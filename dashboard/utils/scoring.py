"""
Combined Scoring Module for Investment Intelligence Dashboard.

Implements a weighted scoring system that combines signals from three intelligence systems:
1. Clinical Trial Signals (40% default weight)
2. Patent Intelligence (30% default weight)
3. Insider/Hiring Signals (30% default weight)

Features:
- Configurable weights for each signal source
- Confidence-adjusted scoring
- Normalization across different score ranges
- Recommendation generation based on combined metrics
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SignalSource(Enum):
    """Enumeration of signal sources."""
    CLINICAL_TRIALS = 'clinical_trials'
    PATENT_INTELLIGENCE = 'patent_intelligence'
    INSIDER_HIRING = 'insider_hiring'


class Recommendation(Enum):
    """Investment recommendation levels."""
    STRONG_BUY = 'STRONG BUY'
    BUY = 'BUY'
    HOLD = 'HOLD'
    SELL = 'SELL'
    STRONG_SELL = 'STRONG SELL'
    NO_DATA = 'NO DATA'


@dataclass
class ScoreWeights:
    """Configuration for signal source weights."""
    clinical_trials: float = 0.40
    patent_intelligence: float = 0.30
    insider_hiring: float = 0.30

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.clinical_trials + self.patent_intelligence + self.insider_hiring
        if not np.isclose(total, 1.0):
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            self.clinical_trials /= total
            self.patent_intelligence /= total
            self.insider_hiring /= total

    def to_dict(self) -> Dict[str, float]:
        """Convert weights to dictionary."""
        return {
            SignalSource.CLINICAL_TRIALS.value: self.clinical_trials,
            SignalSource.PATENT_INTELLIGENCE.value: self.patent_intelligence,
            SignalSource.INSIDER_HIRING.value: self.insider_hiring,
        }


@dataclass
class SignalScore:
    """Individual signal score from a single source."""
    source: SignalSource
    raw_score: float
    normalized_score: float  # 0-1 range
    confidence: float  # 0-1 range
    signal_count: int
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CombinedScore:
    """Combined score from all signal sources."""
    ticker: str
    company_name: str
    combined_score: float  # 0-1 range
    weighted_confidence: float  # 0-1 range
    recommendation: Recommendation
    signal_count: int
    breakdown: Dict[str, SignalScore] = field(default_factory=dict)
    contributing_sources: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame conversion."""
        return {
            'ticker': self.ticker,
            'company_name': self.company_name,
            'combined_score': self.combined_score,
            'confidence': self.weighted_confidence,
            'recommendation': self.recommendation.value,
            'signal_count': self.signal_count,
            'clinical_score': self.breakdown.get(SignalSource.CLINICAL_TRIALS.value, SignalScore(
                SignalSource.CLINICAL_TRIALS, 0, 0, 0, 0
            )).normalized_score if SignalSource.CLINICAL_TRIALS.value in self.breakdown else None,
            'patent_score': self.breakdown.get(SignalSource.PATENT_INTELLIGENCE.value, SignalScore(
                SignalSource.PATENT_INTELLIGENCE, 0, 0, 0, 0
            )).normalized_score if SignalSource.PATENT_INTELLIGENCE.value in self.breakdown else None,
            'insider_score': self.breakdown.get(SignalSource.INSIDER_HIRING.value, SignalScore(
                SignalSource.INSIDER_HIRING, 0, 0, 0, 0
            )).normalized_score if SignalSource.INSIDER_HIRING.value in self.breakdown else None,
            'sources': ', '.join(self.contributing_sources),
        }


class ScoreNormalizer:
    """
    Normalizes scores from different sources to a common 0-1 scale.

    Each source may have different native score ranges:
    - Clinical Trials: 0-10 scale
    - Patent Intelligence: 0-100 scale
    - Insider/Hiring: -10 to +10 scale
    """

    # Score ranges for each source
    SCORE_RANGES = {
        SignalSource.CLINICAL_TRIALS.value: (0, 10),
        SignalSource.PATENT_INTELLIGENCE.value: (0, 100),
        SignalSource.INSIDER_HIRING.value: (-10, 10),
    }

    @classmethod
    def normalize(cls, score: float, source: str) -> float:
        """
        Normalize a score from its native range to 0-1.

        Args:
            score: Raw score value
            source: Signal source identifier

        Returns:
            Normalized score in 0-1 range
        """
        min_val, max_val = cls.SCORE_RANGES.get(source, (0, 1))
        if max_val == min_val:
            return 0.5
        normalized = (score - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))

    @classmethod
    def denormalize(cls, normalized_score: float, source: str) -> float:
        """
        Convert a normalized 0-1 score back to its native range.

        Args:
            normalized_score: Score in 0-1 range
            source: Signal source identifier

        Returns:
            Score in native range
        """
        min_val, max_val = cls.SCORE_RANGES.get(source, (0, 1))
        return min_val + normalized_score * (max_val - min_val)


class RecommendationEngine:
    """
    Generates investment recommendations based on combined scores.

    Uses configurable thresholds for recommendation levels.
    """

    def __init__(
        self,
        strong_buy_threshold: float = 0.75,
        buy_threshold: float = 0.60,
        sell_threshold: float = 0.40,
        strong_sell_threshold: float = 0.25,
        min_confidence: float = 0.50,
    ):
        """
        Initialize recommendation thresholds.

        Args:
            strong_buy_threshold: Score threshold for STRONG BUY
            buy_threshold: Score threshold for BUY
            sell_threshold: Score threshold for SELL
            strong_sell_threshold: Score threshold for STRONG SELL
            min_confidence: Minimum confidence for strong recommendations
        """
        self.strong_buy_threshold = strong_buy_threshold
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.strong_sell_threshold = strong_sell_threshold
        self.min_confidence = min_confidence

    def get_recommendation(
        self,
        score: float,
        confidence: float,
    ) -> Recommendation:
        """
        Generate recommendation based on score and confidence.

        Args:
            score: Combined score (0-1)
            confidence: Confidence level (0-1)

        Returns:
            Recommendation enum value
        """
        # High confidence allows for strong recommendations
        if confidence >= self.min_confidence:
            if score >= self.strong_buy_threshold:
                return Recommendation.STRONG_BUY
            elif score >= self.buy_threshold:
                return Recommendation.BUY
            elif score <= self.strong_sell_threshold:
                return Recommendation.STRONG_SELL
            elif score <= self.sell_threshold:
                return Recommendation.SELL
        else:
            # Lower confidence - moderate recommendations only
            if score >= self.buy_threshold:
                return Recommendation.BUY
            elif score <= self.sell_threshold:
                return Recommendation.SELL

        return Recommendation.HOLD

    def get_signal_strength(self, score: float) -> str:
        """
        Get qualitative signal strength description.

        Args:
            score: Combined score (0-1)

        Returns:
            Signal strength description
        """
        if score >= 0.8:
            return "Very Strong Bullish"
        elif score >= 0.6:
            return "Bullish"
        elif score >= 0.4:
            return "Neutral"
        elif score >= 0.2:
            return "Bearish"
        else:
            return "Very Strong Bearish"


class CombinedScorer:
    """
    Main scoring engine that combines signals from all sources.

    Calculates weighted scores, handles missing data gracefully,
    and generates investment recommendations.
    """

    def __init__(
        self,
        weights: Optional[ScoreWeights] = None,
        recommendation_engine: Optional[RecommendationEngine] = None,
    ):
        """
        Initialize the combined scorer.

        Args:
            weights: Score weights configuration
            recommendation_engine: Recommendation generator
        """
        self.weights = weights or ScoreWeights()
        self.recommendation_engine = recommendation_engine or RecommendationEngine()
        self.normalizer = ScoreNormalizer()

    def calculate_combined_score(
        self,
        ticker: str,
        company_name: str,
        scores: Dict[str, Dict[str, Any]],
    ) -> CombinedScore:
        """
        Calculate combined score from individual source scores.

        Args:
            ticker: Company ticker symbol
            company_name: Company name
            scores: Dictionary mapping source names to score data:
                    {
                        'clinical_trials': {'score': 7.5, 'confidence': 0.8, 'count': 3},
                        'patent_intelligence': {'score': 85, 'confidence': 0.7, 'count': 2},
                        'insider_hiring': {'score': 5.0, 'confidence': 0.9, 'count': 5},
                    }

        Returns:
            CombinedScore object with weighted results
        """
        if not scores:
            return CombinedScore(
                ticker=ticker,
                company_name=company_name,
                combined_score=0.5,
                weighted_confidence=0,
                recommendation=Recommendation.NO_DATA,
                signal_count=0,
            )

        weights_dict = self.weights.to_dict()
        breakdown = {}
        weighted_score_sum = 0
        weighted_confidence_sum = 0
        total_weight = 0
        total_signals = 0
        contributing_sources = []

        for source, data in scores.items():
            if source not in weights_dict:
                continue

            raw_score = data.get('score', 0)
            confidence = data.get('confidence', 0.5)
            signal_count = data.get('count', 1)
            details = data.get('details', {})

            # Normalize the score
            normalized_score = self.normalizer.normalize(raw_score, source)

            # Create signal score
            signal_score = SignalScore(
                source=SignalSource(source),
                raw_score=raw_score,
                normalized_score=normalized_score,
                confidence=confidence,
                signal_count=signal_count,
                details=details,
            )
            breakdown[source] = signal_score

            # Calculate weighted contribution
            weight = weights_dict[source]
            weighted_score_sum += normalized_score * weight * confidence
            weighted_confidence_sum += confidence * weight
            total_weight += weight
            total_signals += signal_count
            contributing_sources.append(source)

        # Normalize by actual weights used (handles missing sources)
        if total_weight > 0:
            combined_score = weighted_score_sum / total_weight
            weighted_confidence = weighted_confidence_sum / total_weight
        else:
            combined_score = 0.5
            weighted_confidence = 0

        # Generate recommendation
        recommendation = self.recommendation_engine.get_recommendation(
            combined_score, weighted_confidence
        )

        return CombinedScore(
            ticker=ticker,
            company_name=company_name,
            combined_score=round(combined_score, 4),
            weighted_confidence=round(weighted_confidence, 4),
            recommendation=recommendation,
            signal_count=total_signals,
            breakdown=breakdown,
            contributing_sources=contributing_sources,
        )

    def calculate_batch_scores(
        self,
        data: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Calculate combined scores for multiple companies.

        Args:
            data: List of dictionaries with company data and source scores

        Returns:
            DataFrame with combined scores for all companies
        """
        results = []
        for item in data:
            ticker = item.get('ticker', '')
            company_name = item.get('company_name', ticker)
            scores = item.get('scores', {})

            combined = self.calculate_combined_score(ticker, company_name, scores)
            results.append(combined.to_dict())

        df = pd.DataFrame(results)

        # Sort by combined score descending
        if not df.empty:
            df = df.sort_values('combined_score', ascending=False).reset_index(drop=True)

        return df

    def get_high_confidence_alerts(
        self,
        scores_df: pd.DataFrame,
        score_threshold: float = 0.70,
        confidence_threshold: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """
        Extract high-confidence alert signals from scored data.

        Args:
            scores_df: DataFrame with combined scores
            score_threshold: Minimum score for bullish alerts (and 1-threshold for bearish)
            confidence_threshold: Minimum confidence level

        Returns:
            List of alert dictionaries
        """
        alerts = []

        if scores_df.empty:
            return alerts

        # Bullish alerts
        bullish = scores_df[
            (scores_df['combined_score'] >= score_threshold) &
            (scores_df['confidence'] >= confidence_threshold)
        ]

        for _, row in bullish.iterrows():
            alerts.append({
                'type': 'BULLISH',
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'combined_score': row['combined_score'],
                'confidence': row['confidence'],
                'recommendation': row['recommendation'],
                'message': f"High confidence bullish signal detected for {row['ticker']}",
                'priority': 'HIGH' if row['combined_score'] >= 0.80 else 'MEDIUM',
            })

        # Bearish alerts
        bearish = scores_df[
            (scores_df['combined_score'] <= (1 - score_threshold)) &
            (scores_df['confidence'] >= confidence_threshold)
        ]

        for _, row in bearish.iterrows():
            alerts.append({
                'type': 'BEARISH',
                'ticker': row['ticker'],
                'company_name': row['company_name'],
                'combined_score': row['combined_score'],
                'confidence': row['confidence'],
                'recommendation': row['recommendation'],
                'message': f"High confidence bearish signal detected for {row['ticker']}",
                'priority': 'HIGH' if row['combined_score'] <= 0.20 else 'MEDIUM',
            })

        # Sort by priority and score
        priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        alerts.sort(key=lambda x: (priority_order.get(x['priority'], 2), -abs(x['combined_score'] - 0.5)))

        return alerts

    def calculate_score_statistics(
        self,
        scores_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Calculate summary statistics for scored data.

        Args:
            scores_df: DataFrame with combined scores

        Returns:
            Dictionary with score statistics
        """
        if scores_df.empty:
            return {
                'total_companies': 0,
                'avg_score': 0,
                'avg_confidence': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'hold_count': 0,
            }

        bullish_recs = ['STRONG BUY', 'BUY']
        bearish_recs = ['STRONG SELL', 'SELL']

        return {
            'total_companies': len(scores_df),
            'avg_score': round(scores_df['combined_score'].mean(), 3),
            'avg_confidence': round(scores_df['confidence'].mean(), 3),
            'max_score': round(scores_df['combined_score'].max(), 3),
            'min_score': round(scores_df['combined_score'].min(), 3),
            'bullish_count': len(scores_df[scores_df['recommendation'].isin(bullish_recs)]),
            'bearish_count': len(scores_df[scores_df['recommendation'].isin(bearish_recs)]),
            'hold_count': len(scores_df[scores_df['recommendation'] == 'HOLD']),
            'high_confidence_count': len(scores_df[scores_df['confidence'] >= 0.70]),
        }

    def calculate_correlation_matrix(
        self,
        scores_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between different score sources.

        Args:
            scores_df: DataFrame with combined scores

        Returns:
            Correlation matrix DataFrame
        """
        score_cols = ['clinical_score', 'patent_score', 'insider_score']
        available_cols = [c for c in score_cols if c in scores_df.columns]

        if len(available_cols) < 2:
            return pd.DataFrame()

        # Fill NaN with mean for correlation calculation
        subset = scores_df[available_cols].fillna(scores_df[available_cols].mean())
        return subset.corr()


# Utility functions for easy import
def calculate_combined_scores(
    data: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Convenience function to calculate combined scores.

    Args:
        data: List of company data with scores
        weights: Optional custom weights

    Returns:
        DataFrame with combined scores
    """
    score_weights = ScoreWeights(**weights) if weights else ScoreWeights()
    scorer = CombinedScorer(weights=score_weights)
    return scorer.calculate_batch_scores(data)


def get_recommendation(score: float, confidence: float) -> str:
    """
    Get recommendation for a single score/confidence pair.

    Args:
        score: Combined score (0-1)
        confidence: Confidence level (0-1)

    Returns:
        Recommendation string
    """
    engine = RecommendationEngine()
    return engine.get_recommendation(score, confidence).value
