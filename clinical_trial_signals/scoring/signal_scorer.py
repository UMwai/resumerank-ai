"""
Signal scoring model for Clinical Trial Signal Detection System.

Implements a multi-factor scoring model to generate composite scores
and investment recommendations based on detected signals.
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database.models import Trial, TrialSignal, TrialScore

logger = logging.getLogger(__name__)


class Recommendation(Enum):
    """Investment recommendations based on score."""
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SHORT = "SHORT"
    STRONG_SHORT = "STRONG_SHORT"


@dataclass
class ScoringResult:
    """Result of scoring a trial."""
    trial_id: str
    composite_score: float
    confidence: float
    recommendation: Recommendation
    signal_count: int
    total_weight: int
    max_possible_weight: int
    contributing_signals: List[Dict[str, Any]]


class SignalScorer:
    """
    Scores clinical trials based on detected signals to generate
    investment recommendations.
    """

    def __init__(self):
        self.config = config.signal
        self.weights = self.config.weights

        # Maximum possible weights for normalization
        self._max_positive_weight = sum(w for w in self.weights.values() if w > 0)
        self._max_negative_weight = abs(sum(w for w in self.weights.values() if w < 0))

    def score_trial(
        self,
        trial_id: str,
        lookback_days: int = 30
    ) -> Optional[ScoringResult]:
        """
        Calculate composite score for a trial based on recent signals.

        Args:
            trial_id: NCT identifier of the trial
            lookback_days: Number of days to consider for signals

        Returns:
            ScoringResult with score, confidence, and recommendation
        """
        # Get recent signals for this trial
        signals = TrialSignal.get_by_trial(trial_id, days=lookback_days)

        if not signals:
            logger.debug(f"No signals found for {trial_id} in last {lookback_days} days")
            return None

        # Calculate total weight
        total_weight = sum(s.signal_weight for s in signals)
        signal_count = len(signals)

        # Normalize score to 0-10 scale
        # Score of 5 is neutral, >5 is positive, <5 is negative
        if total_weight >= 0:
            # Positive score: 5-10 range
            normalized = 5 + (total_weight / self._max_positive_weight * 5)
        else:
            # Negative score: 0-5 range
            normalized = 5 + (total_weight / self._max_negative_weight * 5)

        # Clamp to valid range
        composite_score = max(0, min(10, normalized))

        # Calculate confidence based on signal count and diversity
        confidence = self._calculate_confidence(signals)

        # Determine recommendation
        recommendation = self._get_recommendation(composite_score, confidence)

        # Build contributing signals summary
        contributing = [
            {
                "signal_type": s.signal_type,
                "signal_weight": s.signal_weight,
                "detected_date": s.detected_date.isoformat() if s.detected_date else None,
                "description": s.signal_value,
            }
            for s in signals
        ]

        result = ScoringResult(
            trial_id=trial_id,
            composite_score=round(composite_score, 2),
            confidence=round(confidence, 2),
            recommendation=recommendation,
            signal_count=signal_count,
            total_weight=total_weight,
            max_possible_weight=self._max_positive_weight,
            contributing_signals=contributing
        )

        logger.info(
            f"Scored {trial_id}: {composite_score:.2f} ({recommendation.value}) "
            f"[{signal_count} signals, confidence: {confidence:.2f}]"
        )

        return result

    def _calculate_confidence(self, signals: List[TrialSignal]) -> float:
        """
        Calculate confidence score based on signal quantity and diversity.

        Confidence factors:
        - Number of signals (more = higher confidence)
        - Diversity of signal types (more types = higher confidence)
        - Signal consistency (all pointing same direction = higher confidence)
        - Historical accuracy (from config)

        Returns:
            Confidence score between 0 and 1
        """
        if not signals:
            return 0.0

        signal_count = len(signals)
        unique_types = len(set(s.signal_type for s in signals))

        # Factor 1: Signal count (normalized, max at 6+ signals)
        count_factor = min(signal_count / 6.0, 1.0)

        # Factor 2: Signal type diversity (normalized)
        # More diverse signals give higher confidence
        diversity_factor = min(unique_types / 4.0, 1.0)

        # Factor 3: Signal consistency
        # If all signals point in the same direction, higher confidence
        positive_count = sum(1 for s in signals if s.signal_weight > 0)
        negative_count = sum(1 for s in signals if s.signal_weight < 0)

        if positive_count == 0 or negative_count == 0:
            consistency_factor = 1.0  # All same direction
        else:
            # Mixed signals reduce confidence
            total = positive_count + negative_count
            dominant = max(positive_count, negative_count)
            consistency_factor = dominant / total

        # Factor 4: Historical accuracy (from config/backtesting)
        historical_factor = self.config.historical_accuracy

        # Weighted combination
        confidence = (
            count_factor * 0.25 +
            diversity_factor * 0.2 +
            consistency_factor * 0.25 +
            historical_factor * 0.3
        )

        return min(confidence, 1.0)

    def _get_recommendation(
        self,
        score: float,
        confidence: float
    ) -> Recommendation:
        """
        Determine investment recommendation based on score and confidence.

        From spec:
        - score >= 7 and confidence >= 0.7: STRONG BUY
        - score >= 5: BUY
        - score <= 3 and confidence >= 0.7: SHORT
        - else: HOLD/MONITOR
        """
        confidence_threshold = self.config.confidence_threshold

        if score >= self.config.strong_buy_threshold and confidence >= confidence_threshold:
            return Recommendation.STRONG_BUY
        elif score >= self.config.buy_threshold:
            return Recommendation.BUY
        elif score <= self.config.short_threshold and confidence >= confidence_threshold:
            if score <= 2:
                return Recommendation.STRONG_SHORT
            return Recommendation.SHORT
        else:
            return Recommendation.HOLD

    def score_and_save(
        self,
        trial_id: str,
        lookback_days: int = 30
    ) -> Optional[TrialScore]:
        """
        Score a trial and save the result to the database.

        Args:
            trial_id: NCT identifier
            lookback_days: Number of days to consider

        Returns:
            TrialScore object or None if no signals
        """
        result = self.score_trial(trial_id, lookback_days)

        if not result:
            return None

        score = TrialScore(
            trial_id=result.trial_id,
            composite_score=result.composite_score,
            confidence=result.confidence,
            recommendation=result.recommendation.value,
            score_date=date.today(),
            contributing_signals={
                "signals": result.contributing_signals,
                "total_weight": result.total_weight,
                "signal_count": result.signal_count,
            }
        )
        score.save()

        return score

    def score_all_trials(self, lookback_days: int = 30) -> List[TrialScore]:
        """
        Score all monitored trials.

        Args:
            lookback_days: Number of days to consider for signals

        Returns:
            List of TrialScore objects
        """
        trials = Trial.get_monitored()
        scores = []

        for trial in trials:
            score = self.score_and_save(trial.trial_id, lookback_days)
            if score:
                scores.append(score)

        logger.info(f"Scored {len(scores)} trials out of {len(trials)} monitored")
        return scores

    def get_actionable_opportunities(self) -> List[Dict[str, Any]]:
        """
        Get trials with actionable scores (strong signals).

        Returns:
            List of dictionaries with trial and score info
        """
        return TrialScore.get_actionable()

    def generate_summary(self, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Generate a summary of all trial scores for reporting.

        Returns:
            Dictionary with summary statistics and top opportunities
        """
        trials = Trial.get_monitored()
        scores = []

        for trial in trials:
            result = self.score_trial(trial.trial_id, lookback_days)
            if result:
                scores.append({
                    "trial_id": trial.trial_id,
                    "drug_name": trial.drug_name,
                    "company": trial.company_ticker,
                    "indication": trial.indication,
                    "score": result.composite_score,
                    "confidence": result.confidence,
                    "recommendation": result.recommendation.value,
                    "signal_count": result.signal_count,
                })

        # Sort by score (descending for buys, ascending for shorts)
        scores_sorted = sorted(scores, key=lambda x: x["score"], reverse=True)

        # Identify top opportunities
        strong_buys = [s for s in scores_sorted if s["recommendation"] in ("STRONG_BUY", "BUY")]
        shorts = [s for s in scores_sorted if s["recommendation"] in ("SHORT", "STRONG_SHORT")]

        return {
            "total_trials": len(trials),
            "scored_trials": len(scores),
            "strong_buys": strong_buys[:5],  # Top 5
            "shorts": shorts[:5],  # Top 5
            "all_scores": scores_sorted,
            "generated_at": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # Test scoring module
    logging.basicConfig(level=logging.INFO)

    print("Testing signal scorer...")

    scorer = SignalScorer()

    # Print configuration
    print(f"\nScoring configuration:")
    print(f"  Strong buy threshold: >= {scorer.config.strong_buy_threshold}")
    print(f"  Buy threshold: >= {scorer.config.buy_threshold}")
    print(f"  Short threshold: <= {scorer.config.short_threshold}")
    print(f"  Confidence threshold: {scorer.config.confidence_threshold}")
    print(f"  Max positive weight: {scorer._max_positive_weight}")
    print(f"  Max negative weight: {scorer._max_negative_weight}")

    # Test recommendation logic
    print("\nTesting recommendation logic:")
    test_cases = [
        (8.5, 0.8, "High score, high confidence"),
        (7.0, 0.5, "High score, low confidence"),
        (5.5, 0.6, "Moderate score"),
        (2.5, 0.8, "Low score, high confidence"),
        (3.0, 0.5, "Low score, low confidence"),
    ]

    for score, confidence, description in test_cases:
        rec = scorer._get_recommendation(score, confidence)
        print(f"  {description}: score={score}, conf={confidence} -> {rec.value}")
