"""
Simple Regime Detector Module
==============================
Simplified market regime detection using only VIX and SMA crossovers.
3 regimes only: BULL / BEAR / NEUTRAL
Simple decision tree without complex weighting.

Author: Backend Systems Architect
Date: November 2024
Version: 1.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple, Optional, List
from enum import Enum
import logging
from dataclasses import dataclass, field

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleRegime(Enum):
    """Simplified market regime classifications."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"


@dataclass
class SimpleIndicators:
    """
    Simplified regime indicators using only VIX and SMA.
    """
    # Price data
    current_price: float
    sma_50: float
    sma_200: float

    # VIX data
    vix_level: float
    vix_ma20: float

    # Calculated indicators
    sma_crossover: str  # "golden" / "death" / "neutral"
    price_trend: str    # "above_both" / "below_both" / "between"
    vix_regime: str     # "low" / "medium" / "high"

    # Timestamp
    date: datetime = field(default_factory=datetime.now)


class SimpleRegimeDetector:
    """
    Simplified regime detection using VIX levels and SMA crossovers.
    Decision tree approach without complex weighting.
    """

    # VIX thresholds (simplified)
    VIX_LOW = 20       # Below 20: Low volatility (bullish)
    VIX_HIGH = 25      # Above 25: High volatility (bearish)

    # SMA trend thresholds
    TREND_THRESHOLD = 0.02  # 2% above/below for confirmation

    def __init__(self):
        """Initialize simple regime detector."""
        self.current_regime = SimpleRegime.NEUTRAL
        self.regime_history = []
        self.indicator_history = []
        logger.info("Simple Regime Detector initialized")

    def detect_regime(
        self,
        price_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        date: Optional[datetime] = None
    ) -> Tuple[SimpleRegime, float, SimpleIndicators]:
        """
        Detect market regime using simple decision tree.

        Args:
            price_data: DataFrame with price and SMA data
            vix_data: DataFrame with VIX data
            date: Date to detect regime for (defaults to latest)

        Returns:
            Tuple of (regime, confidence, indicators)
        """
        # Get data for specific date or latest
        if date is None:
            price_row = price_data.iloc[-1]
            vix_row = vix_data.iloc[-1]
            date = price_data.index[-1]
        else:
            price_row = price_data.loc[date]
            vix_row = vix_data.loc[date]

        # Calculate indicators
        indicators = self._calculate_indicators(price_row, vix_row, date)

        # Apply simple decision tree
        regime, confidence = self._apply_decision_tree(indicators)

        # Store history
        self.current_regime = regime
        self.regime_history.append({
            'date': date,
            'regime': regime,
            'confidence': confidence
        })
        self.indicator_history.append(indicators)

        logger.info(f"Date: {date.date()}, Regime: {regime.value.upper()}, Confidence: {confidence:.1%}")

        return regime, confidence, indicators

    def _calculate_indicators(
        self,
        price_row: pd.Series,
        vix_row: pd.Series,
        date: datetime
    ) -> SimpleIndicators:
        """Calculate simplified indicators."""

        # Extract values
        current_price = price_row['close']
        sma_50 = price_row.get('sma_50', price_row.get('sma_20', current_price))
        sma_200 = price_row.get('sma_200', price_row.get('sma_50', current_price))
        vix_level = vix_row['close']
        vix_ma20 = vix_row.get('vix_ma20', vix_level)

        # Determine SMA crossover
        if sma_50 > sma_200 * (1 + self.TREND_THRESHOLD):
            sma_crossover = "golden"  # Golden cross (bullish)
        elif sma_50 < sma_200 * (1 - self.TREND_THRESHOLD):
            sma_crossover = "death"   # Death cross (bearish)
        else:
            sma_crossover = "neutral"

        # Determine price trend
        if current_price > sma_50 and current_price > sma_200:
            price_trend = "above_both"
        elif current_price < sma_50 and current_price < sma_200:
            price_trend = "below_both"
        else:
            price_trend = "between"

        # Determine VIX regime
        if vix_level < self.VIX_LOW:
            vix_regime = "low"
        elif vix_level > self.VIX_HIGH:
            vix_regime = "high"
        else:
            vix_regime = "medium"

        return SimpleIndicators(
            current_price=current_price,
            sma_50=sma_50,
            sma_200=sma_200,
            vix_level=vix_level,
            vix_ma20=vix_ma20,
            sma_crossover=sma_crossover,
            price_trend=price_trend,
            vix_regime=vix_regime,
            date=date
        )

    def _apply_decision_tree(self, indicators: SimpleIndicators) -> Tuple[SimpleRegime, float]:
        """
        Apply simple decision tree for regime detection.

        Decision Tree Logic:
        1. VIX High (>25) → Check trend → BEAR or NEUTRAL
        2. VIX Low (<20) → Check trend → BULL or NEUTRAL
        3. VIX Medium → Check SMA crossover and price position

        Returns:
            Tuple of (regime, confidence)
        """
        regime = SimpleRegime.NEUTRAL
        confidence = 0.5

        # DECISION TREE

        # Branch 1: High VIX (bearish bias)
        if indicators.vix_regime == "high":
            if indicators.price_trend == "below_both":
                # High VIX + Price below SMAs = BEAR
                regime = SimpleRegime.BEAR
                confidence = 0.9
            elif indicators.sma_crossover == "death":
                # High VIX + Death cross = BEAR
                regime = SimpleRegime.BEAR
                confidence = 0.85
            elif indicators.price_trend == "above_both":
                # High VIX but price strong = NEUTRAL (conflicting signals)
                regime = SimpleRegime.NEUTRAL
                confidence = 0.6
            else:
                # High VIX, mixed signals = NEUTRAL
                regime = SimpleRegime.NEUTRAL
                confidence = 0.65

        # Branch 2: Low VIX (bullish bias)
        elif indicators.vix_regime == "low":
            if indicators.price_trend == "above_both":
                # Low VIX + Price above SMAs = BULL
                regime = SimpleRegime.BULL
                confidence = 0.9
            elif indicators.sma_crossover == "golden":
                # Low VIX + Golden cross = BULL
                regime = SimpleRegime.BULL
                confidence = 0.85
            elif indicators.price_trend == "below_both":
                # Low VIX but price weak = NEUTRAL (conflicting signals)
                regime = SimpleRegime.NEUTRAL
                confidence = 0.6
            else:
                # Low VIX, mixed signals = slight BULL bias
                regime = SimpleRegime.BULL
                confidence = 0.7

        # Branch 3: Medium VIX (use trend indicators)
        else:  # vix_regime == "medium"
            if indicators.sma_crossover == "golden" and indicators.price_trend == "above_both":
                # Golden cross + Price above = BULL
                regime = SimpleRegime.BULL
                confidence = 0.8
            elif indicators.sma_crossover == "death" and indicators.price_trend == "below_both":
                # Death cross + Price below = BEAR
                regime = SimpleRegime.BEAR
                confidence = 0.8
            elif indicators.price_trend == "above_both":
                # Price strength = slight BULL
                regime = SimpleRegime.BULL
                confidence = 0.65
            elif indicators.price_trend == "below_both":
                # Price weakness = slight BEAR
                regime = SimpleRegime.BEAR
                confidence = 0.65
            else:
                # True neutral conditions
                regime = SimpleRegime.NEUTRAL
                confidence = 0.7

        return regime, confidence

    def get_regime_stats(self) -> Dict:
        """
        Get statistics about regime detection.

        Returns:
            Dictionary with regime statistics
        """
        if not self.regime_history:
            return {}

        df = pd.DataFrame(self.regime_history)

        stats = {
            'total_days': len(df),
            'current_regime': self.current_regime.value,
            'regime_counts': df['regime'].value_counts().to_dict(),
            'regime_percentages': (df['regime'].value_counts() / len(df) * 100).to_dict(),
            'average_confidence': df['confidence'].mean(),
            'regime_changes': 0,
            'last_change': None
        }

        # Count regime changes
        for i in range(1, len(df)):
            if df.iloc[i]['regime'] != df.iloc[i-1]['regime']:
                stats['regime_changes'] += 1
                stats['last_change'] = df.iloc[i]['date']

        return stats

    def validate_on_historical(
        self,
        price_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        known_periods: Dict[str, List[Tuple[str, str]]]
    ) -> Dict:
        """
        Validate regime detection on historical periods.

        Args:
            price_data: Historical price data with SMAs
            vix_data: Historical VIX data
            known_periods: Dictionary of known regime periods
                          e.g., {'bear': [('2020-02-20', '2020-03-23'), ...]}

        Returns:
            Validation metrics
        """
        results = {
            'total_accuracy': 0,
            'regime_accuracy': {},
            'period_results': [],
            'confusion_matrix': {
                'bull': {'bull': 0, 'bear': 0, 'neutral': 0},
                'bear': {'bull': 0, 'bear': 0, 'neutral': 0},
                'neutral': {'bull': 0, 'bear': 0, 'neutral': 0}
            }
        }

        total_correct = 0
        total_days = 0

        # Validate each known period
        for true_regime, periods in known_periods.items():
            regime_correct = 0
            regime_total = 0

            for start_str, end_str in periods:
                start = pd.to_datetime(start_str)
                end = pd.to_datetime(end_str)

                # Get data for period
                period_price = price_data[start:end]
                period_vix = vix_data[start:end]

                if period_price.empty or period_vix.empty:
                    logger.warning(f"No data for period {start_str} to {end_str}")
                    continue

                period_results = []

                # Detect regime for each day in period
                for date in period_price.index:
                    if date not in period_vix.index:
                        continue

                    detected_regime, confidence, _ = self.detect_regime(
                        period_price[:date],
                        period_vix[:date],
                        date
                    )

                    # Check if correct
                    is_correct = detected_regime.value == true_regime
                    if is_correct:
                        regime_correct += 1
                        total_correct += 1

                    regime_total += 1
                    total_days += 1

                    # Update confusion matrix
                    results['confusion_matrix'][true_regime][detected_regime.value] += 1

                    period_results.append({
                        'date': date,
                        'true_regime': true_regime,
                        'detected_regime': detected_regime.value,
                        'confidence': confidence,
                        'correct': is_correct
                    })

                # Store period results
                period_accuracy = regime_correct / regime_total if regime_total > 0 else 0
                results['period_results'].append({
                    'period': f"{start_str} to {end_str}",
                    'true_regime': true_regime,
                    'accuracy': period_accuracy,
                    'days_analyzed': regime_total,
                    'details': period_results
                })

                logger.info(f"Period {start_str} to {end_str} ({true_regime}): "
                           f"Accuracy {period_accuracy:.1%}")

            # Calculate regime-specific accuracy
            if regime_total > 0:
                results['regime_accuracy'][true_regime] = regime_correct / regime_total

        # Calculate overall accuracy
        if total_days > 0:
            results['total_accuracy'] = total_correct / total_days

        # Calculate precision and recall for each regime
        results['metrics'] = self._calculate_metrics(results['confusion_matrix'])

        return results

    def _calculate_metrics(self, confusion_matrix: Dict) -> Dict:
        """Calculate precision, recall, and F1 score from confusion matrix."""
        metrics = {}

        for regime in ['bull', 'bear', 'neutral']:
            # True positives
            tp = confusion_matrix[regime][regime]

            # False positives (predicted as regime but actually other)
            fp = sum(confusion_matrix[other][regime]
                    for other in ['bull', 'bear', 'neutral']
                    if other != regime)

            # False negatives (actually regime but predicted as other)
            fn = sum(confusion_matrix[regime][other]
                    for other in ['bull', 'bear', 'neutral']
                    if other != regime)

            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics[regime] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': tp + fn  # Total actual instances
            }

        return metrics


def create_regime_report(detector: SimpleRegimeDetector, validation_results: Dict) -> str:
    """
    Create a formatted report of regime detection results.

    Args:
        detector: SimpleRegimeDetector instance
        validation_results: Results from validation

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("SIMPLE REGIME DETECTION VALIDATION REPORT")
    report.append("=" * 60)
    report.append("")

    # Overall accuracy
    report.append("OVERALL PERFORMANCE")
    report.append("-" * 30)
    report.append(f"Total Accuracy: {validation_results['total_accuracy']:.1%}")
    report.append("")

    # Regime-specific accuracy
    report.append("ACCURACY BY REGIME")
    report.append("-" * 30)
    for regime, accuracy in validation_results['regime_accuracy'].items():
        report.append(f"{regime.upper()}: {accuracy:.1%}")
    report.append("")

    # Detailed metrics
    report.append("DETAILED METRICS")
    report.append("-" * 30)
    for regime, metrics in validation_results['metrics'].items():
        report.append(f"\n{regime.upper()}:")
        report.append(f"  Precision: {metrics['precision']:.1%}")
        report.append(f"  Recall: {metrics['recall']:.1%}")
        report.append(f"  F1 Score: {metrics['f1_score']:.1%}")
        report.append(f"  Support: {metrics['support']} days")

    # Period analysis
    report.append("\nPERIOD ANALYSIS")
    report.append("-" * 30)
    for period in validation_results['period_results']:
        report.append(f"\n{period['period']} ({period['true_regime'].upper()}):")
        report.append(f"  Accuracy: {period['accuracy']:.1%}")
        report.append(f"  Days Analyzed: {period['days_analyzed']}")

    # Confusion Matrix
    report.append("\nCONFUSION MATRIX")
    report.append("-" * 30)
    report.append("         Predicted")
    report.append("Actual   Bull  Bear  Neutral")
    cm = validation_results['confusion_matrix']
    for actual in ['bull', 'bear', 'neutral']:
        row = f"{actual:8}"
        for predicted in ['bull', 'bear', 'neutral']:
            row += f" {cm[actual][predicted]:5}"
        report.append(row)

    # Regime statistics
    stats = detector.get_regime_stats()
    if stats:
        report.append("\nREGIME STATISTICS")
        report.append("-" * 30)
        report.append(f"Total Days Analyzed: {stats['total_days']}")
        report.append(f"Current Regime: {stats['current_regime'].upper()}")
        report.append(f"Regime Changes: {stats['regime_changes']}")
        report.append(f"Average Confidence: {stats['average_confidence']:.1%}")

        if 'regime_percentages' in stats:
            report.append("\nTime in Each Regime:")
            for regime, pct in stats['regime_percentages'].items():
                report.append(f"  {regime.value.upper()}: {pct:.1%}")

    report.append("\n" + "=" * 60)
    report.append("END OF REPORT")
    report.append("=" * 60)

    return "\n".join(report)


def main():
    """Example usage of SimpleRegimeDetector."""
    detector = SimpleRegimeDetector()

    # Example indicators
    print("Example regime detection:")

    # Bullish scenario
    bull_price = pd.Series({'close': 450, 'sma_50': 440, 'sma_200': 430})
    bull_vix = pd.Series({'close': 15, 'vix_ma20': 16})

    regime, confidence, indicators = detector.detect_regime(
        pd.DataFrame([bull_price]),
        pd.DataFrame([bull_vix])
    )
    print(f"Bullish scenario: {regime.value} (confidence: {confidence:.1%})")

    # Bearish scenario
    bear_price = pd.Series({'close': 400, 'sma_50': 420, 'sma_200': 430})
    bear_vix = pd.Series({'close': 30, 'vix_ma20': 28})

    regime, confidence, indicators = detector.detect_regime(
        pd.DataFrame([bear_price]),
        pd.DataFrame([bear_vix])
    )
    print(f"Bearish scenario: {regime.value} (confidence: {confidence:.1%})")


if __name__ == "__main__":
    main()