"""
Signal Scoring Model for Insider Activity + Hiring Signals System

Aggregates multiple signals (insider trades, institutional holdings, job postings)
into a composite score with time-decay weighting.
"""

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class Signal:
    """Represents a single signal contributing to the score."""
    company_ticker: str
    signal_date: date
    category: str  # 'insider', 'institutional', 'hiring', 'sentiment', 'executive'
    signal_type: str  # e.g., 'CEO_PURCHASE', 'FUND_NEW_POSITION', 'COMMERCIAL_HIRING'
    description: str
    raw_weight: int
    days_ago: int = 0
    decay_factor: float = 1.0
    weighted_score: float = 0.0
    source_id: Optional[int] = None
    source_table: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class SignalScore:
    """Composite signal score for a company."""
    company_ticker: str
    score_date: date
    composite_score: float
    confidence: float
    signal_count: int
    insider_score: float
    institutional_score: float
    hiring_score: float
    sentiment_score: float
    recommendation: str
    contributing_signals: List[Signal]


class SignalScorer:
    """
    Calculates composite signal scores for biotech companies.

    The scoring model:
    1. Collects signals from multiple sources (insider trades, 13F, jobs)
    2. Applies time-decay weighting (recent signals matter more)
    3. Aggregates into a composite score (-10 to +10)
    4. Calculates confidence based on signal count and quality
    5. Generates a recommendation (STRONG BUY to STRONG SELL)
    """

    # Recommendation thresholds
    RECOMMENDATION_THRESHOLDS = {
        'STRONG BUY': (6.0, 0.7),     # score >= 6, confidence >= 0.7
        'BUY': (3.0, 0.5),             # score >= 3, confidence >= 0.5
        'STRONG SELL': (-6.0, 0.7),   # score <= -6, confidence >= 0.7
        'SELL': (-3.0, 0.5),           # score <= -3, confidence >= 0.5
    }

    # Signal type definitions for different categories
    INSIDER_SIGNAL_TYPES = {
        'CEO_PURCHASE': {'weight': 5, 'description': 'CEO bought shares'},
        'CEO_SALE': {'weight': -4, 'description': 'CEO sold shares'},
        'CFO_PURCHASE': {'weight': 4, 'description': 'CFO bought shares'},
        'CFO_SALE': {'weight': -5, 'description': 'CFO sold shares'},
        'CMO_PURCHASE': {'weight': 5, 'description': 'CMO bought shares'},
        'CMO_SALE': {'weight': -5, 'description': 'CMO sold shares'},
        'DIRECTOR_PURCHASE': {'weight': 3, 'description': 'Director bought shares'},
        'OFFICER_PURCHASE': {'weight': 3, 'description': 'Officer bought shares'},
        'MULTIPLE_INSIDER_BUY': {'weight': 6, 'description': 'Multiple insiders bought in 30 days'},
        'MULTIPLE_INSIDER_SELL': {'weight': -6, 'description': 'Multiple insiders sold in 30 days'},
    }

    INSTITUTIONAL_SIGNAL_TYPES = {
        'FUND_NEW_POSITION': {'weight': 5, 'description': 'Top fund initiated new position'},
        'FUND_INCREASE_50': {'weight': 4, 'description': 'Top fund increased position >50%'},
        'FUND_INCREASE_25': {'weight': 2, 'description': 'Top fund increased position >25%'},
        'FUND_DECREASE_50': {'weight': -4, 'description': 'Top fund decreased position >50%'},
        'FUND_DECREASE_25': {'weight': -2, 'description': 'Top fund decreased position >25%'},
        'FUND_EXIT': {'weight': -5, 'description': 'Top fund exited position completely'},
        'MULTIPLE_FUNDS_INITIATE': {'weight': 6, 'description': '3+ funds initiated positions'},
        'MULTIPLE_FUNDS_EXIT': {'weight': -6, 'description': '3+ funds exited positions'},
    }

    HIRING_SIGNAL_TYPES = {
        'COMMERCIAL_BUILDUP': {'weight': 5, 'description': '5+ commercial roles posted'},
        'VP_MANUFACTURING': {'weight': 4, 'description': 'VP of Manufacturing hired'},
        'REGULATORY_EXPANSION': {'weight': 3, 'description': 'Regulatory team expansion'},
        'CLINICAL_EXPANSION': {'weight': 4, 'description': 'Clinical operations expansion'},
        'MSL_HIRING': {'weight': 5, 'description': 'Medical Science Liaisons hired'},
        'HIRING_FREEZE': {'weight': -4, 'description': 'Multiple job postings removed'},
    }

    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.lookback_days = self.config.signals.get('lookback_days', 90)
        self.time_decay_halflife = self.config.signals.get('time_decay_halflife', 30)
        self.min_transaction_value = self.config.signals.get('min_transaction_value', 50000)

    def calculate_decay_factor(self, days_ago: int) -> float:
        """
        Calculate time-decay factor using exponential decay.

        Formula: decay = 1 / (1 + days_ago / halflife)

        This gives:
        - Today (0 days): 1.0
        - Half-life (30 days): 0.5
        - 60 days: 0.33
        - 90 days: 0.25
        """
        return 1.0 / (1.0 + days_ago / self.time_decay_halflife)

    def get_insider_signals(self, ticker: str, lookback_days: int = None) -> List[Signal]:
        """
        Get insider trading signals for a company.

        Queries insider_transactions table and generates signals based on:
        - Transaction type (buy/sell)
        - Insider role (CEO, CFO, CMO, Director)
        - Transaction value
        - Timing patterns (multiple insiders)
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        cutoff_date = date.today() - timedelta(days=lookback_days)

        # Get relevant transactions
        transactions = self.db.execute("""
            SELECT
                transaction_id,
                company_ticker,
                insider_name,
                insider_title,
                transaction_date,
                transaction_type,
                transaction_value,
                is_10b5_1_plan,
                signal_weight,
                is_director,
                is_officer
            FROM insider_transactions
            WHERE company_ticker = %s
            AND transaction_date >= %s
            AND is_10b5_1_plan = FALSE
            AND transaction_value >= %s
            ORDER BY transaction_date DESC
        """, (ticker, cutoff_date, self.min_transaction_value))

        signals = []

        # Track for multiple insider detection
        buyers_30d = set()
        sellers_30d = set()
        recent_cutoff = date.today() - timedelta(days=30)

        for trans in transactions:
            days_ago = (date.today() - trans['transaction_date']).days
            decay_factor = self.calculate_decay_factor(days_ago)

            title = (trans['insider_title'] or '').upper()
            is_purchase = trans['transaction_type'] == 'Purchase'

            # Determine signal type
            signal_type = None
            if 'CEO' in title or 'CHIEF EXECUTIVE' in title:
                signal_type = 'CEO_PURCHASE' if is_purchase else 'CEO_SALE'
            elif 'CFO' in title or 'CHIEF FINANCIAL' in title:
                signal_type = 'CFO_PURCHASE' if is_purchase else 'CFO_SALE'
            elif 'CMO' in title or 'CHIEF MEDICAL' in title:
                signal_type = 'CMO_PURCHASE' if is_purchase else 'CMO_SALE'
            elif trans['is_director']:
                signal_type = 'DIRECTOR_PURCHASE' if is_purchase else None
            elif trans['is_officer']:
                signal_type = 'OFFICER_PURCHASE' if is_purchase else None

            if signal_type and signal_type in self.INSIDER_SIGNAL_TYPES:
                signal_def = self.INSIDER_SIGNAL_TYPES[signal_type]
                weighted_score = signal_def['weight'] * decay_factor

                signals.append(Signal(
                    company_ticker=ticker,
                    signal_date=trans['transaction_date'],
                    category='insider',
                    signal_type=signal_type,
                    description=f"{trans['insider_name']} ({trans['insider_title']}): ${trans['transaction_value']:,.0f}",
                    raw_weight=signal_def['weight'],
                    days_ago=days_ago,
                    decay_factor=decay_factor,
                    weighted_score=weighted_score,
                    source_id=trans['transaction_id'],
                    source_table='insider_transactions',
                    metadata={'transaction_value': float(trans['transaction_value'])}
                ))

            # Track for multiple insider signal
            if trans['transaction_date'] >= recent_cutoff:
                if is_purchase:
                    buyers_30d.add(trans['insider_name'])
                else:
                    sellers_30d.add(trans['insider_name'])

        # Check for multiple insider signals
        if len(buyers_30d) >= 2:
            signal_def = self.INSIDER_SIGNAL_TYPES['MULTIPLE_INSIDER_BUY']
            signals.append(Signal(
                company_ticker=ticker,
                signal_date=date.today(),
                category='insider',
                signal_type='MULTIPLE_INSIDER_BUY',
                description=f"{len(buyers_30d)} insiders bought in last 30 days",
                raw_weight=signal_def['weight'],
                days_ago=0,
                decay_factor=1.0,
                weighted_score=signal_def['weight'],
                metadata={'insider_count': len(buyers_30d)}
            ))

        if len(sellers_30d) >= 2:
            signal_def = self.INSIDER_SIGNAL_TYPES['MULTIPLE_INSIDER_SELL']
            signals.append(Signal(
                company_ticker=ticker,
                signal_date=date.today(),
                category='insider',
                signal_type='MULTIPLE_INSIDER_SELL',
                description=f"{len(sellers_30d)} insiders sold in last 30 days",
                raw_weight=signal_def['weight'],
                days_ago=0,
                decay_factor=1.0,
                weighted_score=signal_def['weight'],
                metadata={'insider_count': len(sellers_30d)}
            ))

        return signals

    def get_institutional_signals(self, ticker: str, lookback_days: int = None) -> List[Signal]:
        """
        Get institutional investor signals for a company.

        Queries institutional_holdings table for:
        - New positions by top funds
        - Significant position increases/decreases
        - Complete exits
        - Multiple fund convergence
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        cutoff_date = date.today() - timedelta(days=lookback_days)

        holdings = self.db.execute("""
            SELECT
                holding_id,
                fund_name,
                company_ticker,
                quarter_end,
                shares,
                pct_change_shares,
                is_new_position,
                is_exit,
                signal_weight
            FROM institutional_holdings
            WHERE company_ticker = %s
            AND quarter_end >= %s
            ORDER BY quarter_end DESC
        """, (ticker, cutoff_date))

        signals = []

        # Track for multiple fund convergence
        new_positions = []
        exits = []

        for holding in holdings:
            days_ago = (date.today() - holding['quarter_end']).days
            decay_factor = self.calculate_decay_factor(days_ago)

            signal_type = None
            pct_change = holding['pct_change_shares'] or 0

            if holding['is_new_position']:
                signal_type = 'FUND_NEW_POSITION'
                new_positions.append(holding['fund_name'])
            elif holding['is_exit']:
                signal_type = 'FUND_EXIT'
                exits.append(holding['fund_name'])
            elif pct_change > 50:
                signal_type = 'FUND_INCREASE_50'
            elif pct_change > 25:
                signal_type = 'FUND_INCREASE_25'
            elif pct_change < -50:
                signal_type = 'FUND_DECREASE_50'
            elif pct_change < -25:
                signal_type = 'FUND_DECREASE_25'

            if signal_type and signal_type in self.INSTITUTIONAL_SIGNAL_TYPES:
                signal_def = self.INSTITUTIONAL_SIGNAL_TYPES[signal_type]
                weighted_score = signal_def['weight'] * decay_factor

                signals.append(Signal(
                    company_ticker=ticker,
                    signal_date=holding['quarter_end'],
                    category='institutional',
                    signal_type=signal_type,
                    description=f"{holding['fund_name']}: {pct_change:+.1f}% change",
                    raw_weight=signal_def['weight'],
                    days_ago=days_ago,
                    decay_factor=decay_factor,
                    weighted_score=weighted_score,
                    source_id=holding['holding_id'],
                    source_table='institutional_holdings',
                    metadata={'pct_change': float(pct_change)}
                ))

        # Multiple fund convergence signals
        if len(new_positions) >= 3:
            signal_def = self.INSTITUTIONAL_SIGNAL_TYPES['MULTIPLE_FUNDS_INITIATE']
            signals.append(Signal(
                company_ticker=ticker,
                signal_date=date.today(),
                category='institutional',
                signal_type='MULTIPLE_FUNDS_INITIATE',
                description=f"{len(new_positions)} funds initiated new positions",
                raw_weight=signal_def['weight'],
                days_ago=0,
                decay_factor=1.0,
                weighted_score=signal_def['weight'],
                metadata={'fund_count': len(new_positions)}
            ))

        if len(exits) >= 3:
            signal_def = self.INSTITUTIONAL_SIGNAL_TYPES['MULTIPLE_FUNDS_EXIT']
            signals.append(Signal(
                company_ticker=ticker,
                signal_date=date.today(),
                category='institutional',
                signal_type='MULTIPLE_FUNDS_EXIT',
                description=f"{len(exits)} funds exited positions",
                raw_weight=signal_def['weight'],
                days_ago=0,
                decay_factor=1.0,
                weighted_score=signal_def['weight'],
                metadata={'fund_count': len(exits)}
            ))

        return signals

    def get_hiring_signals(self, ticker: str, lookback_days: int = None) -> List[Signal]:
        """
        Get hiring pattern signals for a company.

        Queries job_postings table for:
        - Commercial role buildup (launch preparation)
        - Manufacturing scale-up
        - Clinical operations expansion
        - Hiring freezes (job removals)
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        cutoff_date = date.today() - timedelta(days=lookback_days)

        # Get job posting counts by category
        job_stats = self.db.execute_one("""
            SELECT
                COUNT(*) as total_jobs,
                SUM(CASE WHEN is_commercial_role THEN 1 ELSE 0 END) as commercial_jobs,
                SUM(CASE WHEN is_manufacturing_role THEN 1 ELSE 0 END) as manufacturing_jobs,
                SUM(CASE WHEN is_clinical_role THEN 1 ELSE 0 END) as clinical_jobs,
                SUM(CASE WHEN is_rd_role THEN 1 ELSE 0 END) as rd_jobs,
                SUM(CASE WHEN is_senior_role AND is_manufacturing_role THEN 1 ELSE 0 END) as senior_manufacturing
            FROM job_postings
            WHERE company_ticker = %s
            AND first_seen_date >= %s
            AND removal_date IS NULL
        """, (ticker, cutoff_date))

        signals = []

        if job_stats:
            # Commercial buildup
            if (job_stats.get('commercial_jobs') or 0) >= 5:
                signal_def = self.HIRING_SIGNAL_TYPES['COMMERCIAL_BUILDUP']
                signals.append(Signal(
                    company_ticker=ticker,
                    signal_date=date.today(),
                    category='hiring',
                    signal_type='COMMERCIAL_BUILDUP',
                    description=f"{job_stats['commercial_jobs']} commercial roles posted",
                    raw_weight=signal_def['weight'],
                    days_ago=0,
                    decay_factor=1.0,
                    weighted_score=signal_def['weight'],
                    metadata={'job_count': job_stats['commercial_jobs']}
                ))

            # VP Manufacturing
            if (job_stats.get('senior_manufacturing') or 0) >= 1:
                signal_def = self.HIRING_SIGNAL_TYPES['VP_MANUFACTURING']
                signals.append(Signal(
                    company_ticker=ticker,
                    signal_date=date.today(),
                    category='hiring',
                    signal_type='VP_MANUFACTURING',
                    description="Senior manufacturing roles posted",
                    raw_weight=signal_def['weight'],
                    days_ago=0,
                    decay_factor=1.0,
                    weighted_score=signal_def['weight']
                ))

            # Clinical expansion
            if (job_stats.get('clinical_jobs') or 0) >= 3:
                signal_def = self.HIRING_SIGNAL_TYPES['CLINICAL_EXPANSION']
                signals.append(Signal(
                    company_ticker=ticker,
                    signal_date=date.today(),
                    category='hiring',
                    signal_type='CLINICAL_EXPANSION',
                    description=f"{job_stats['clinical_jobs']} clinical operations roles",
                    raw_weight=signal_def['weight'],
                    days_ago=0,
                    decay_factor=1.0,
                    weighted_score=signal_def['weight'],
                    metadata={'job_count': job_stats['clinical_jobs']}
                ))

        # Check for job removals (hiring freeze signal)
        removals = self.db.execute_one("""
            SELECT COUNT(*) as removal_count
            FROM job_postings
            WHERE company_ticker = %s
            AND removal_date >= %s
        """, (ticker, cutoff_date))

        if removals and (removals.get('removal_count') or 0) >= 5:
            signal_def = self.HIRING_SIGNAL_TYPES['HIRING_FREEZE']
            signals.append(Signal(
                company_ticker=ticker,
                signal_date=date.today(),
                category='hiring',
                signal_type='HIRING_FREEZE',
                description=f"{removals['removal_count']} job postings removed",
                raw_weight=signal_def['weight'],
                days_ago=0,
                decay_factor=1.0,
                weighted_score=signal_def['weight'],
                metadata={'removal_count': removals['removal_count']}
            ))

        return signals

    def calculate_score(self, ticker: str, lookback_days: int = None) -> SignalScore:
        """
        Calculate composite signal score for a company.

        Aggregates all signals with time-decay weighting and produces:
        - Composite score (-10 to +10)
        - Confidence (0 to 1)
        - Recommendation
        """
        if lookback_days is None:
            lookback_days = self.lookback_days

        # Collect all signals
        insider_signals = self.get_insider_signals(ticker, lookback_days)
        institutional_signals = self.get_institutional_signals(ticker, lookback_days)
        hiring_signals = self.get_hiring_signals(ticker, lookback_days)

        all_signals = insider_signals + institutional_signals + hiring_signals

        # Calculate category scores
        insider_score = sum(s.weighted_score for s in insider_signals)
        institutional_score = sum(s.weighted_score for s in institutional_signals)
        hiring_score = sum(s.weighted_score for s in hiring_signals)
        sentiment_score = 0.0  # Reserved for future Glassdoor integration

        # Total raw score
        total_score = sum(s.weighted_score for s in all_signals)

        # Normalize to -10 to +10 scale
        # Divide by 2 as suggested in spec, then clamp
        normalized_score = max(-10.0, min(10.0, total_score / 2.0))

        # Calculate confidence based on signal quality
        # Higher confidence when we have more signals and they're recent
        if all_signals:
            total_weight = sum(abs(s.raw_weight) * s.decay_factor for s in all_signals)
            confidence = min(1.0, total_weight / 20.0)
        else:
            confidence = 0.0

        # Determine recommendation
        recommendation = self._get_recommendation(normalized_score, confidence)

        return SignalScore(
            company_ticker=ticker,
            score_date=date.today(),
            composite_score=round(normalized_score, 2),
            confidence=round(confidence, 2),
            signal_count=len(all_signals),
            insider_score=round(insider_score, 2),
            institutional_score=round(institutional_score, 2),
            hiring_score=round(hiring_score, 2),
            sentiment_score=round(sentiment_score, 2),
            recommendation=recommendation,
            contributing_signals=all_signals
        )

    def _get_recommendation(self, score: float, confidence: float) -> str:
        """Determine recommendation based on score and confidence."""
        if score >= 6.0 and confidence >= 0.7:
            return "STRONG BUY"
        elif score >= 3.0 and confidence >= 0.5:
            return "BUY"
        elif score <= -6.0 and confidence >= 0.7:
            return "STRONG SELL"
        elif score <= -3.0 and confidence >= 0.5:
            return "SELL"
        else:
            return "NEUTRAL"

    def save_score(self, score: SignalScore) -> int:
        """Save signal score to database."""
        # Convert signals to JSON-serializable format
        signals_json = [
            {
                'signal_date': str(s.signal_date),
                'category': s.category,
                'signal_type': s.signal_type,
                'description': s.description,
                'raw_weight': s.raw_weight,
                'days_ago': s.days_ago,
                'decay_factor': round(s.decay_factor, 4),
                'weighted_score': round(s.weighted_score, 3),
                'metadata': s.metadata
            }
            for s in score.contributing_signals
        ]

        data = {
            'company_ticker': score.company_ticker,
            'score_date': score.score_date,
            'composite_score': score.composite_score,
            'confidence': score.confidence,
            'signal_count': score.signal_count,
            'insider_score': score.insider_score,
            'institutional_score': score.institutional_score,
            'hiring_score': score.hiring_score,
            'sentiment_score': score.sentiment_score,
            'recommendation': score.recommendation,
            'contributing_signals': signals_json,
            'lookback_days': self.lookback_days
        }

        try:
            score_id = self.db.upsert(
                'signal_scores',
                data,
                conflict_columns=['company_ticker', 'score_date'],
                update_columns=[
                    'composite_score', 'confidence', 'signal_count',
                    'insider_score', 'institutional_score', 'hiring_score',
                    'sentiment_score', 'recommendation', 'contributing_signals'
                ]
            )
            logger.info(f"Saved score for {score.company_ticker}: {score.composite_score} ({score.recommendation})")
            return score_id
        except Exception as e:
            logger.error(f"Failed to save score for {score.company_ticker}: {e}")
            return None

    def score_all_companies(self, tickers: Optional[List[str]] = None) -> List[SignalScore]:
        """
        Calculate and save scores for all watchlist companies.

        Args:
            tickers: Optional list of tickers (defaults to watchlist)

        Returns:
            List of SignalScore objects
        """
        if tickers is None:
            tickers = self.config.watchlist

        scores = []

        for ticker in tickers:
            try:
                score = self.calculate_score(ticker)
                self.save_score(score)
                scores.append(score)
            except Exception as e:
                logger.error(f"Failed to score {ticker}: {e}")

        # Sort by composite score (descending)
        scores.sort(key=lambda s: s.composite_score, reverse=True)

        return scores

    def get_top_signals(self, n_bullish: int = 5,
                        n_bearish: int = 5) -> Tuple[List[SignalScore], List[SignalScore]]:
        """
        Get top bullish and bearish signal scores.

        Returns:
            Tuple of (top_bullish_scores, top_bearish_scores)
        """
        scores = self.score_all_companies()

        # Filter for significant scores
        bullish = [s for s in scores if s.composite_score >= 3.0 and s.confidence >= 0.5]
        bearish = [s for s in scores if s.composite_score <= -3.0 and s.confidence >= 0.5]

        # Sort and take top N
        bullish.sort(key=lambda s: (s.composite_score, s.confidence), reverse=True)
        bearish.sort(key=lambda s: (s.composite_score, s.confidence))

        return bullish[:n_bullish], bearish[:n_bearish]


if __name__ == '__main__':
    # Test the scorer
    scorer = SignalScorer()

    # Calculate scores for test companies
    test_tickers = ['MRNA', 'VRTX', 'CRSP']

    for ticker in test_tickers:
        score = scorer.calculate_score(ticker)
        print(f"\n{ticker}: Score={score.composite_score:.2f}, "
              f"Confidence={score.confidence:.2f}, Rec={score.recommendation}")
        print(f"  Signals: {score.signal_count}")
        print(f"  Insider: {score.insider_score:.2f}")
        print(f"  Institutional: {score.institutional_score:.2f}")
        print(f"  Hiring: {score.hiring_score:.2f}")

        for signal in score.contributing_signals[:5]:
            print(f"    - {signal.signal_type}: {signal.description} "
                  f"(weight={signal.weighted_score:.2f})")
