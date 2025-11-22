"""
Pattern Detection Algorithms for Insider/Hiring Signals

Identifies high-value patterns in signal data:
- Insider buying clusters (3+ insiders in 30 days)
- Smart money convergence (2+ top funds buying)
- Hiring surge patterns (5+ jobs in 2 weeks)
- Executive exodus (2+ departures in 90 days)
- Automatic alerts for significant patterns
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


class PatternType(Enum):
    """Types of detectable patterns."""
    INSIDER_CLUSTER_BUY = "insider_cluster_buy"
    INSIDER_CLUSTER_SELL = "insider_cluster_sell"
    SMART_MONEY_CONVERGENCE = "smart_money_convergence"
    SMART_MONEY_EXIT = "smart_money_exit"
    HIRING_SURGE = "hiring_surge"
    HIRING_FREEZE = "hiring_freeze"
    EXECUTIVE_EXODUS = "executive_exodus"
    EXECUTIVE_HIRING = "executive_hiring"
    SENTIMENT_SHIFT = "sentiment_shift"
    CATALYST_PREPARATION = "catalyst_preparation"
    DISTRESS_SIGNAL = "distress_signal"


class AlertPriority(Enum):
    """Priority levels for alerts."""
    CRITICAL = "critical"  # Immediate attention needed
    HIGH = "high"         # Important, review within hours
    MEDIUM = "medium"     # Review within a day
    LOW = "low"           # Informational


@dataclass
class PatternAlert:
    """Represents a detected pattern alert."""
    pattern_type: PatternType
    ticker: str
    company_name: Optional[str]
    detected_at: datetime
    priority: AlertPriority
    confidence: float  # 0.0 to 1.0
    signal_strength: float  # -10 to +10
    description: str
    details: Dict[str, Any]
    contributing_signals: List[Dict]
    recommended_action: str
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'pattern_type': self.pattern_type.value,
            'ticker': self.ticker,
            'company_name': self.company_name,
            'detected_at': self.detected_at.isoformat(),
            'priority': self.priority.value,
            'confidence': self.confidence,
            'signal_strength': self.signal_strength,
            'description': self.description,
            'details': self.details,
            'contributing_signals': self.contributing_signals,
            'recommended_action': self.recommended_action,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class InsiderActivity:
    """Represents insider trading activity."""
    ticker: str
    insider_name: str
    insider_title: str
    transaction_date: date
    transaction_type: str  # 'buy' or 'sell'
    shares: int
    value: float
    is_10b5_1: bool = False


@dataclass
class InstitutionalActivity:
    """Represents institutional investor activity."""
    ticker: str
    fund_name: str
    quarter_end: date
    action: str  # 'new_position', 'increase', 'decrease', 'exit'
    shares: int
    value: float
    pct_change: float


@dataclass
class JobActivity:
    """Represents job posting activity."""
    ticker: str
    job_title: str
    department: str
    posted_date: date
    is_senior: bool
    is_commercial: bool
    is_clinical: bool


@dataclass
class ExecutiveChange:
    """Represents executive change activity."""
    ticker: str
    executive_name: str
    title: str
    change_type: str  # 'departure', 'hiring', 'promotion'
    effective_date: date
    is_voluntary: Optional[bool]


class PatternDetector:
    """
    Detects significant patterns in insider activity, institutional holdings,
    job postings, and executive changes.

    Implements pattern detection algorithms optimized for biotech sector.
    """

    # Pattern thresholds
    THRESHOLDS = {
        'insider_cluster_count': 3,        # Min insiders for cluster
        'insider_cluster_days': 30,        # Window for insider cluster
        'smart_money_count': 2,            # Min funds for convergence
        'smart_money_days': 90,            # Window for fund activity
        'hiring_surge_count': 5,           # Min jobs for surge
        'hiring_surge_days': 14,           # Window for hiring surge
        'executive_exodus_count': 2,       # Min departures for exodus
        'executive_exodus_days': 90,       # Window for departures
        'min_transaction_value': 50000,    # Min value for insider signal
    }

    # Signal weights for different patterns
    PATTERN_WEIGHTS = {
        PatternType.INSIDER_CLUSTER_BUY: 8,
        PatternType.INSIDER_CLUSTER_SELL: -7,
        PatternType.SMART_MONEY_CONVERGENCE: 7,
        PatternType.SMART_MONEY_EXIT: -8,
        PatternType.HIRING_SURGE: 6,
        PatternType.HIRING_FREEZE: -5,
        PatternType.EXECUTIVE_EXODUS: -8,
        PatternType.EXECUTIVE_HIRING: 4,
        PatternType.SENTIMENT_SHIFT: 3,  # Can be positive or negative
        PatternType.CATALYST_PREPARATION: 7,
        PatternType.DISTRESS_SIGNAL: -9,
    }

    # Top biotech funds to track
    TOP_FUNDS = [
        'OrbiMed Advisors',
        'Baker Brothers Advisors',
        'RA Capital Management',
        'Perceptive Advisors',
        'Deerfield Management',
        'Farallon Capital',
        'RTW Investments',
        'Tang Capital',
        'Cormorant Asset Management',
        'Great Point Partners',
    ]

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pattern detector."""
        self.config = get_config(config_path) if config_path else None
        self.alerts: List[PatternAlert] = []
        self._alert_history: Dict[str, datetime] = {}  # Deduplication

    def detect_all_patterns(
        self,
        ticker: str,
        insider_activities: List[InsiderActivity],
        institutional_activities: List[InstitutionalActivity],
        job_activities: List[JobActivity],
        executive_changes: List[ExecutiveChange],
        glassdoor_sentiment: Optional[Dict] = None,
        company_name: Optional[str] = None
    ) -> List[PatternAlert]:
        """
        Detect all patterns for a company.

        Args:
            ticker: Company ticker
            insider_activities: Recent insider transactions
            institutional_activities: Recent institutional changes
            job_activities: Recent job postings
            executive_changes: Recent executive changes
            glassdoor_sentiment: Optional sentiment data
            company_name: Company name for alerts

        Returns:
            List of PatternAlert objects
        """
        alerts = []

        # Insider patterns
        insider_alerts = self.detect_insider_cluster(
            ticker, insider_activities, company_name
        )
        alerts.extend(insider_alerts)

        # Institutional patterns
        institutional_alerts = self.detect_smart_money_convergence(
            ticker, institutional_activities, company_name
        )
        alerts.extend(institutional_alerts)

        # Hiring patterns
        hiring_alerts = self.detect_hiring_patterns(
            ticker, job_activities, company_name
        )
        alerts.extend(hiring_alerts)

        # Executive patterns
        executive_alerts = self.detect_executive_patterns(
            ticker, executive_changes, company_name
        )
        alerts.extend(executive_alerts)

        # Combined patterns (catalyst preparation, distress)
        combined_alerts = self.detect_combined_patterns(
            ticker,
            insider_activities,
            institutional_activities,
            job_activities,
            executive_changes,
            glassdoor_sentiment,
            company_name
        )
        alerts.extend(combined_alerts)

        # Deduplicate and store
        new_alerts = self._deduplicate_alerts(alerts)
        self.alerts.extend(new_alerts)

        return new_alerts

    def detect_insider_cluster(
        self,
        ticker: str,
        activities: List[InsiderActivity],
        company_name: Optional[str] = None
    ) -> List[PatternAlert]:
        """
        Detect insider buying/selling clusters.

        A cluster is 3+ distinct insiders trading in the same direction
        within 30 days. Excludes 10b5-1 plan trades.
        """
        alerts = []

        if not activities:
            return alerts

        # Filter out 10b5-1 trades and low-value transactions
        significant = [
            a for a in activities
            if not a.is_10b5_1
            and a.value >= self.THRESHOLDS['min_transaction_value']
        ]

        # Group by direction
        buys = [a for a in significant if a.transaction_type.lower() == 'buy']
        sells = [a for a in significant if a.transaction_type.lower() == 'sell']

        # Check for buy cluster
        buy_cluster = self._check_cluster(buys, self.THRESHOLDS['insider_cluster_days'])
        if buy_cluster:
            insiders, total_value = buy_cluster
            alert = PatternAlert(
                pattern_type=PatternType.INSIDER_CLUSTER_BUY,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.HIGH,
                confidence=min(0.95, 0.6 + len(insiders) * 0.1),
                signal_strength=self.PATTERN_WEIGHTS[PatternType.INSIDER_CLUSTER_BUY],
                description=f"{len(insiders)} insiders bought ${total_value:,.0f} worth in 30 days",
                details={
                    'insider_count': len(insiders),
                    'total_value': total_value,
                    'insiders': list(insiders),
                    'window_days': self.THRESHOLDS['insider_cluster_days'],
                },
                contributing_signals=[
                    {
                        'insider': a.insider_name,
                        'title': a.insider_title,
                        'value': a.value,
                        'date': str(a.transaction_date),
                    }
                    for a in buys if a.insider_name in insiders
                ],
                recommended_action="Strong bullish signal. Consider initiating or adding to position.",
                expires_at=datetime.now() + timedelta(days=30)
            )
            alerts.append(alert)

        # Check for sell cluster
        sell_cluster = self._check_cluster(sells, self.THRESHOLDS['insider_cluster_days'])
        if sell_cluster:
            insiders, total_value = sell_cluster
            alert = PatternAlert(
                pattern_type=PatternType.INSIDER_CLUSTER_SELL,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.HIGH,
                confidence=min(0.9, 0.5 + len(insiders) * 0.1),
                signal_strength=self.PATTERN_WEIGHTS[PatternType.INSIDER_CLUSTER_SELL],
                description=f"{len(insiders)} insiders sold ${total_value:,.0f} worth in 30 days",
                details={
                    'insider_count': len(insiders),
                    'total_value': total_value,
                    'insiders': list(insiders),
                    'window_days': self.THRESHOLDS['insider_cluster_days'],
                },
                contributing_signals=[
                    {
                        'insider': a.insider_name,
                        'title': a.insider_title,
                        'value': a.value,
                        'date': str(a.transaction_date),
                    }
                    for a in sells if a.insider_name in insiders
                ],
                recommended_action="Bearish signal. Review position and consider reducing exposure.",
                expires_at=datetime.now() + timedelta(days=30)
            )
            alerts.append(alert)

        return alerts

    def _check_cluster(
        self,
        activities: List[InsiderActivity],
        window_days: int
    ) -> Optional[Tuple[set, float]]:
        """Check if activities form a cluster within the window."""
        if len(activities) < self.THRESHOLDS['insider_cluster_count']:
            return None

        # Sort by date
        sorted_acts = sorted(activities, key=lambda a: a.transaction_date)

        # Sliding window to find cluster
        for i, start_act in enumerate(sorted_acts):
            window_end = start_act.transaction_date + timedelta(days=window_days)

            # Find all activities in window
            window_acts = [
                a for a in sorted_acts[i:]
                if a.transaction_date <= window_end
            ]

            # Count unique insiders
            insiders = set(a.insider_name for a in window_acts)

            if len(insiders) >= self.THRESHOLDS['insider_cluster_count']:
                total_value = sum(a.value for a in window_acts)
                return insiders, total_value

        return None

    def detect_smart_money_convergence(
        self,
        ticker: str,
        activities: List[InstitutionalActivity],
        company_name: Optional[str] = None
    ) -> List[PatternAlert]:
        """
        Detect smart money convergence or exit patterns.

        Convergence: 2+ top funds initiating or significantly increasing positions.
        Exit: 2+ top funds exiting or significantly decreasing positions.
        """
        alerts = []

        if not activities:
            return alerts

        # Filter to top funds
        top_fund_activities = [
            a for a in activities
            if any(fund.lower() in a.fund_name.lower() for fund in self.TOP_FUNDS)
        ]

        # Check for convergence (new positions or >25% increases)
        bullish = [
            a for a in top_fund_activities
            if a.action in ['new_position', 'increase'] and a.pct_change > 25
        ]

        fund_names_bullish = set(a.fund_name for a in bullish)
        if len(fund_names_bullish) >= self.THRESHOLDS['smart_money_count']:
            total_value = sum(a.value for a in bullish)
            alert = PatternAlert(
                pattern_type=PatternType.SMART_MONEY_CONVERGENCE,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.HIGH,
                confidence=min(0.9, 0.6 + len(fund_names_bullish) * 0.1),
                signal_strength=self.PATTERN_WEIGHTS[PatternType.SMART_MONEY_CONVERGENCE],
                description=f"{len(fund_names_bullish)} top biotech funds accumulating",
                details={
                    'fund_count': len(fund_names_bullish),
                    'funds': list(fund_names_bullish),
                    'total_value': total_value,
                },
                contributing_signals=[
                    {
                        'fund': a.fund_name,
                        'action': a.action,
                        'pct_change': a.pct_change,
                        'quarter': str(a.quarter_end),
                    }
                    for a in bullish
                ],
                recommended_action="Smart money accumulation. Strong buy signal.",
                expires_at=datetime.now() + timedelta(days=90)
            )
            alerts.append(alert)

        # Check for exit pattern
        bearish = [
            a for a in top_fund_activities
            if a.action in ['exit', 'decrease'] and (a.pct_change < -25 or a.action == 'exit')
        ]

        fund_names_bearish = set(a.fund_name for a in bearish)
        if len(fund_names_bearish) >= self.THRESHOLDS['smart_money_count']:
            alert = PatternAlert(
                pattern_type=PatternType.SMART_MONEY_EXIT,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.CRITICAL,
                confidence=min(0.95, 0.7 + len(fund_names_bearish) * 0.1),
                signal_strength=self.PATTERN_WEIGHTS[PatternType.SMART_MONEY_EXIT],
                description=f"{len(fund_names_bearish)} top biotech funds exiting/reducing",
                details={
                    'fund_count': len(fund_names_bearish),
                    'funds': list(fund_names_bearish),
                },
                contributing_signals=[
                    {
                        'fund': a.fund_name,
                        'action': a.action,
                        'pct_change': a.pct_change,
                        'quarter': str(a.quarter_end),
                    }
                    for a in bearish
                ],
                recommended_action="Major red flag. Consider exiting position.",
                expires_at=datetime.now() + timedelta(days=90)
            )
            alerts.append(alert)

        return alerts

    def detect_hiring_patterns(
        self,
        ticker: str,
        activities: List[JobActivity],
        company_name: Optional[str] = None
    ) -> List[PatternAlert]:
        """
        Detect hiring surge or freeze patterns.

        Surge: 5+ jobs posted in 2 weeks, especially commercial/clinical.
        Freeze: Significant reduction in job postings.
        """
        alerts = []

        if not activities:
            return alerts

        window_days = self.THRESHOLDS['hiring_surge_days']
        cutoff_date = date.today() - timedelta(days=window_days)

        recent_jobs = [a for a in activities if a.posted_date >= cutoff_date]

        # Check for hiring surge
        if len(recent_jobs) >= self.THRESHOLDS['hiring_surge_count']:
            commercial_count = sum(1 for j in recent_jobs if j.is_commercial)
            clinical_count = sum(1 for j in recent_jobs if j.is_clinical)
            senior_count = sum(1 for j in recent_jobs if j.is_senior)

            # Higher confidence if commercial roles (launch prep)
            confidence = 0.6 + min(0.3, commercial_count * 0.05 + clinical_count * 0.03)

            pattern_type = PatternType.HIRING_SURGE
            signal_strength = self.PATTERN_WEIGHTS[PatternType.HIRING_SURGE]

            # Extra weight for commercial buildup
            if commercial_count >= 3:
                signal_strength += 2
                pattern_type = PatternType.CATALYST_PREPARATION

            alert = PatternAlert(
                pattern_type=pattern_type,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.MEDIUM if commercial_count < 3 else AlertPriority.HIGH,
                confidence=confidence,
                signal_strength=signal_strength,
                description=f"{len(recent_jobs)} jobs posted in {window_days} days",
                details={
                    'total_jobs': len(recent_jobs),
                    'commercial_roles': commercial_count,
                    'clinical_roles': clinical_count,
                    'senior_roles': senior_count,
                    'window_days': window_days,
                },
                contributing_signals=[
                    {
                        'title': j.job_title,
                        'department': j.department,
                        'is_commercial': j.is_commercial,
                        'posted': str(j.posted_date),
                    }
                    for j in recent_jobs[:10]  # Limit to 10
                ],
                recommended_action=(
                    "Commercial buildup suggests launch preparation. Strong bullish signal."
                    if commercial_count >= 3 else
                    "Hiring surge indicates growth. Moderately bullish."
                ),
                expires_at=datetime.now() + timedelta(days=30)
            )
            alerts.append(alert)

        return alerts

    def detect_executive_patterns(
        self,
        ticker: str,
        changes: List[ExecutiveChange],
        company_name: Optional[str] = None
    ) -> List[PatternAlert]:
        """
        Detect executive exodus or strategic hiring patterns.

        Exodus: 2+ executive departures in 90 days.
        Strategic hiring: Key executive additions (CMO, CCO, etc.)
        """
        alerts = []

        if not changes:
            return alerts

        window_days = self.THRESHOLDS['executive_exodus_days']
        cutoff_date = date.today() - timedelta(days=window_days)

        recent_changes = [c for c in changes if c.effective_date >= cutoff_date]

        # Departures
        departures = [c for c in recent_changes if c.change_type == 'departure']
        if len(departures) >= self.THRESHOLDS['executive_exodus_count']:
            # Check for C-suite departures
            c_suite = [d for d in departures if 'chief' in d.title.lower() or d.title.upper().startswith('C')]

            priority = AlertPriority.CRITICAL if len(c_suite) >= 1 else AlertPriority.HIGH

            alert = PatternAlert(
                pattern_type=PatternType.EXECUTIVE_EXODUS,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=priority,
                confidence=min(0.9, 0.6 + len(departures) * 0.1),
                signal_strength=self.PATTERN_WEIGHTS[PatternType.EXECUTIVE_EXODUS],
                description=f"{len(departures)} executive departures in {window_days} days",
                details={
                    'departure_count': len(departures),
                    'c_suite_departures': len(c_suite),
                    'executives': [d.executive_name for d in departures],
                },
                contributing_signals=[
                    {
                        'name': d.executive_name,
                        'title': d.title,
                        'date': str(d.effective_date),
                        'voluntary': d.is_voluntary,
                    }
                    for d in departures
                ],
                recommended_action="Major red flag. Leadership instability suggests problems.",
                expires_at=datetime.now() + timedelta(days=90)
            )
            alerts.append(alert)

        # Strategic hires
        hires = [c for c in recent_changes if c.change_type == 'hiring']
        strategic_titles = ['chief commercial', 'chief medical', 'head of commercial',
                           'vp commercial', 'chief operating']

        strategic_hires = [
            h for h in hires
            if any(title in h.title.lower() for title in strategic_titles)
        ]

        if strategic_hires:
            alert = PatternAlert(
                pattern_type=PatternType.EXECUTIVE_HIRING,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.MEDIUM,
                confidence=0.7,
                signal_strength=self.PATTERN_WEIGHTS[PatternType.EXECUTIVE_HIRING],
                description=f"Strategic executive hire: {strategic_hires[0].title}",
                details={
                    'hire_count': len(strategic_hires),
                    'executives': [(h.executive_name, h.title) for h in strategic_hires],
                },
                contributing_signals=[
                    {
                        'name': h.executive_name,
                        'title': h.title,
                        'date': str(h.effective_date),
                    }
                    for h in strategic_hires
                ],
                recommended_action="Strategic hire suggests upcoming milestone. Bullish signal.",
                expires_at=datetime.now() + timedelta(days=60)
            )
            alerts.append(alert)

        return alerts

    def detect_combined_patterns(
        self,
        ticker: str,
        insider_activities: List[InsiderActivity],
        institutional_activities: List[InstitutionalActivity],
        job_activities: List[JobActivity],
        executive_changes: List[ExecutiveChange],
        glassdoor_sentiment: Optional[Dict],
        company_name: Optional[str] = None
    ) -> List[PatternAlert]:
        """
        Detect combined patterns that span multiple signal types.

        Catalyst Preparation: Insider buying + hiring surge + commercial buildup
        Distress Signal: Executive exodus + hiring freeze + negative sentiment
        """
        alerts = []

        # Count positive and negative signals
        positive_signals = 0
        negative_signals = 0
        signal_details = []

        # Insider signals
        if insider_activities:
            buys = [a for a in insider_activities if a.transaction_type.lower() == 'buy' and not a.is_10b5_1]
            sells = [a for a in insider_activities if a.transaction_type.lower() == 'sell' and not a.is_10b5_1]

            if len(buys) >= 2:
                positive_signals += 2
                signal_details.append(f"{len(buys)} insider purchases")
            if len(sells) >= 3:
                negative_signals += 2
                signal_details.append(f"{len(sells)} insider sales")

        # Institutional signals
        if institutional_activities:
            top_fund_buys = [
                a for a in institutional_activities
                if a.action in ['new_position', 'increase']
                and any(f.lower() in a.fund_name.lower() for f in self.TOP_FUNDS)
            ]
            top_fund_exits = [
                a for a in institutional_activities
                if a.action == 'exit'
                and any(f.lower() in a.fund_name.lower() for f in self.TOP_FUNDS)
            ]

            if top_fund_buys:
                positive_signals += len(top_fund_buys)
                signal_details.append(f"{len(top_fund_buys)} fund accumulations")
            if top_fund_exits:
                negative_signals += len(top_fund_exits) * 2
                signal_details.append(f"{len(top_fund_exits)} fund exits")

        # Job signals
        if job_activities:
            recent_jobs = [
                j for j in job_activities
                if j.posted_date >= date.today() - timedelta(days=30)
            ]
            commercial_jobs = [j for j in recent_jobs if j.is_commercial]

            if len(commercial_jobs) >= 3:
                positive_signals += 3
                signal_details.append(f"{len(commercial_jobs)} commercial hires")

        # Executive signals
        if executive_changes:
            recent = [
                c for c in executive_changes
                if c.effective_date >= date.today() - timedelta(days=90)
            ]
            departures = [c for c in recent if c.change_type == 'departure']
            if len(departures) >= 2:
                negative_signals += 3
                signal_details.append(f"{len(departures)} executive departures")

        # Sentiment
        if glassdoor_sentiment:
            avg_sentiment = glassdoor_sentiment.get('avg_sentiment', 0)
            if avg_sentiment > 0.3:
                positive_signals += 1
            elif avg_sentiment < -0.3:
                negative_signals += 2
                signal_details.append("Negative employee sentiment")

        # Generate combined alerts
        if positive_signals >= 5 and negative_signals <= 1:
            alert = PatternAlert(
                pattern_type=PatternType.CATALYST_PREPARATION,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.HIGH,
                confidence=min(0.95, 0.6 + positive_signals * 0.05),
                signal_strength=9,
                description="Multiple bullish signals converging - potential catalyst ahead",
                details={
                    'positive_signals': positive_signals,
                    'negative_signals': negative_signals,
                    'signal_breakdown': signal_details,
                },
                contributing_signals=[{'signal': s} for s in signal_details],
                recommended_action=(
                    "Strong multi-signal convergence. High conviction buy opportunity. "
                    "Consider larger position size."
                ),
                expires_at=datetime.now() + timedelta(days=60)
            )
            alerts.append(alert)

        elif negative_signals >= 5:
            alert = PatternAlert(
                pattern_type=PatternType.DISTRESS_SIGNAL,
                ticker=ticker,
                company_name=company_name,
                detected_at=datetime.now(),
                priority=AlertPriority.CRITICAL,
                confidence=min(0.95, 0.6 + negative_signals * 0.05),
                signal_strength=-9,
                description="Multiple distress signals detected - significant risk",
                details={
                    'positive_signals': positive_signals,
                    'negative_signals': negative_signals,
                    'signal_breakdown': signal_details,
                },
                contributing_signals=[{'signal': s} for s in signal_details],
                recommended_action=(
                    "Multiple red flags. Exit or significantly reduce position immediately."
                ),
                expires_at=datetime.now() + timedelta(days=30)
            )
            alerts.append(alert)

        return alerts

    def _deduplicate_alerts(self, alerts: List[PatternAlert]) -> List[PatternAlert]:
        """Remove duplicate alerts based on type, ticker, and time."""
        new_alerts = []

        for alert in alerts:
            key = f"{alert.pattern_type.value}_{alert.ticker}"

            # Check if we've seen this alert recently
            last_seen = self._alert_history.get(key)

            if last_seen is None or (datetime.now() - last_seen).days >= 1:
                new_alerts.append(alert)
                self._alert_history[key] = datetime.now()

        return new_alerts

    def get_active_alerts(
        self,
        priority_filter: Optional[AlertPriority] = None,
        ticker_filter: Optional[str] = None
    ) -> List[PatternAlert]:
        """Get active (non-expired) alerts with optional filters."""
        now = datetime.now()

        active = [
            a for a in self.alerts
            if a.expires_at is None or a.expires_at > now
        ]

        if priority_filter:
            active = [a for a in active if a.priority == priority_filter]

        if ticker_filter:
            active = [a for a in active if a.ticker == ticker_filter]

        return sorted(active, key=lambda a: (a.priority.value, -a.signal_strength))

    def generate_alert_summary(self) -> str:
        """Generate a text summary of active alerts."""
        active = self.get_active_alerts()

        if not active:
            return "No active pattern alerts."

        lines = ["PATTERN ALERT SUMMARY", "=" * 50, ""]

        # Group by priority
        for priority in [AlertPriority.CRITICAL, AlertPriority.HIGH,
                        AlertPriority.MEDIUM, AlertPriority.LOW]:
            priority_alerts = [a for a in active if a.priority == priority]

            if priority_alerts:
                lines.append(f"\n{priority.value.upper()} PRIORITY ({len(priority_alerts)} alerts)")
                lines.append("-" * 40)

                for alert in priority_alerts:
                    lines.append(f"\n  {alert.ticker}: {alert.pattern_type.value}")
                    lines.append(f"    {alert.description}")
                    lines.append(f"    Signal: {alert.signal_strength:+.1f}, Confidence: {alert.confidence:.0%}")
                    lines.append(f"    Action: {alert.recommended_action}")

        return "\n".join(lines)


# Convenience function to run pattern detection
def detect_patterns_for_company(
    ticker: str,
    insider_data: List[Dict],
    institutional_data: List[Dict],
    job_data: List[Dict],
    executive_data: List[Dict],
    sentiment_data: Optional[Dict] = None
) -> List[PatternAlert]:
    """
    Convenience function to detect patterns for a company.

    Converts raw data dicts to dataclass objects and runs detection.
    """
    detector = PatternDetector()

    # Convert dicts to dataclass objects
    insider_activities = [
        InsiderActivity(
            ticker=d.get('ticker', ticker),
            insider_name=d.get('insider_name', ''),
            insider_title=d.get('insider_title', ''),
            transaction_date=d.get('transaction_date', date.today()),
            transaction_type=d.get('transaction_type', ''),
            shares=d.get('shares', 0),
            value=d.get('value', 0),
            is_10b5_1=d.get('is_10b5_1', False),
        )
        for d in insider_data
    ]

    institutional_activities = [
        InstitutionalActivity(
            ticker=d.get('ticker', ticker),
            fund_name=d.get('fund_name', ''),
            quarter_end=d.get('quarter_end', date.today()),
            action=d.get('action', ''),
            shares=d.get('shares', 0),
            value=d.get('value', 0),
            pct_change=d.get('pct_change', 0),
        )
        for d in institutional_data
    ]

    job_activities = [
        JobActivity(
            ticker=d.get('ticker', ticker),
            job_title=d.get('job_title', ''),
            department=d.get('department', ''),
            posted_date=d.get('posted_date', date.today()),
            is_senior=d.get('is_senior', False),
            is_commercial=d.get('is_commercial', False),
            is_clinical=d.get('is_clinical', False),
        )
        for d in job_data
    ]

    executive_changes = [
        ExecutiveChange(
            ticker=d.get('ticker', ticker),
            executive_name=d.get('executive_name', ''),
            title=d.get('title', ''),
            change_type=d.get('change_type', ''),
            effective_date=d.get('effective_date', date.today()),
            is_voluntary=d.get('is_voluntary'),
        )
        for d in executive_data
    ]

    return detector.detect_all_patterns(
        ticker=ticker,
        insider_activities=insider_activities,
        institutional_activities=institutional_activities,
        job_activities=job_activities,
        executive_changes=executive_changes,
        glassdoor_sentiment=sentiment_data,
        company_name=None
    )


if __name__ == '__main__':
    # Test pattern detection
    detector = PatternDetector()

    # Sample data
    test_insider = [
        InsiderActivity('MRNA', 'John Doe', 'CEO', date(2024, 1, 15), 'buy', 10000, 150000, False),
        InsiderActivity('MRNA', 'Jane Smith', 'CFO', date(2024, 1, 20), 'buy', 5000, 75000, False),
        InsiderActivity('MRNA', 'Bob Wilson', 'Director', date(2024, 1, 25), 'buy', 3000, 45000, False),
    ]

    test_institutional = [
        InstitutionalActivity('MRNA', 'Baker Brothers Advisors', date(2024, 3, 31),
                             'new_position', 1000000, 150000000, 100),
        InstitutionalActivity('MRNA', 'RA Capital Management', date(2024, 3, 31),
                             'increase', 500000, 75000000, 50),
    ]

    test_jobs = [
        JobActivity('MRNA', 'VP Commercial', 'Commercial', date(2024, 1, 10), True, True, False),
        JobActivity('MRNA', 'MSL Lead', 'Medical', date(2024, 1, 12), True, True, False),
        JobActivity('MRNA', 'Sales Rep', 'Commercial', date(2024, 1, 14), False, True, False),
        JobActivity('MRNA', 'Sales Rep', 'Commercial', date(2024, 1, 15), False, True, False),
        JobActivity('MRNA', 'Marketing Manager', 'Commercial', date(2024, 1, 16), False, True, False),
    ]

    alerts = detector.detect_all_patterns(
        'MRNA',
        test_insider,
        test_institutional,
        test_jobs,
        [],
        company_name='Moderna'
    )

    print(detector.generate_alert_summary())

    for alert in alerts:
        print(f"\n{alert.pattern_type.value}: {alert.description}")
        print(f"  Priority: {alert.priority.value}")
        print(f"  Signal: {alert.signal_strength:+.1f}")
        print(f"  Action: {alert.recommended_action}")
