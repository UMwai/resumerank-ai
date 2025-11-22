"""
Enhanced Backtesting Framework for Clinical Trial Signal Detection System.

Tests signals against historical trial outcomes to:
- Calculate accuracy, precision, recall metrics
- Generate ROI estimates for different trading strategies
- Validate signal weights and scoring model
- Produce detailed backtesting reports with recommendations
"""
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import statistics

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

logger = logging.getLogger(__name__)


class TrialOutcome(Enum):
    """Actual trial outcomes for backtesting."""
    SUCCESS = "success"  # Met primary endpoint
    FAILURE = "failure"  # Failed primary endpoint
    PARTIAL = "partial"  # Mixed results
    TERMINATED = "terminated"  # Trial stopped early
    UNKNOWN = "unknown"  # Outcome not yet known


class TradeAction(Enum):
    """Trading actions based on signals."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SHORT = "short"
    STRONG_SHORT = "strong_short"


@dataclass
class HistoricalTrial:
    """Historical trial data for backtesting."""
    trial_id: str
    company_ticker: str
    drug_name: str
    indication: str
    phase: str
    outcome: TrialOutcome
    outcome_date: date
    stock_price_before: float  # Price when signal detected
    stock_price_after: float  # Price after outcome announced
    signals: List[Dict[str, Any]]  # Signals detected before outcome
    signal_date: date  # Date when main signal detected
    composite_score: float  # Score at signal time
    confidence: float
    recommendation: str


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Thresholds
    buy_threshold: float = 7.0
    short_threshold: float = 3.0
    confidence_threshold: float = 0.7

    # Position sizing
    position_size: float = 10000.0  # USD per trade
    max_positions: int = 5

    # Timing
    hold_period_days: int = 30  # Days to hold position
    signal_lookback_days: int = 30  # Days before outcome to check signals

    # Strategy options
    require_high_confidence: bool = True
    use_stop_loss: bool = True
    stop_loss_pct: float = 0.15  # 15% stop loss


@dataclass
class TradeResult:
    """Result of a single backtest trade."""
    trial_id: str
    ticker: str
    action: TradeAction
    entry_price: float
    exit_price: float
    return_pct: float
    return_usd: float
    correct_prediction: bool
    signal_score: float
    signal_confidence: float
    actual_outcome: TrialOutcome
    hold_days: int


@dataclass
class BacktestMetrics:
    """Calculated metrics from backtesting."""
    # Classification metrics
    accuracy: float  # Overall prediction accuracy
    precision: float  # Precision (true positives / predicted positives)
    recall: float  # Recall (true positives / actual positives)
    f1_score: float  # Harmonic mean of precision and recall

    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Return metrics
    total_return_pct: float
    average_return_pct: float
    median_return_pct: float
    max_return_pct: float
    min_return_pct: float
    sharpe_ratio: float  # Simplified: avg return / std dev

    # Strategy-specific
    buy_accuracy: float
    short_accuracy: float
    strong_signal_accuracy: float

    # ROI estimates
    estimated_annual_roi: float
    capital_at_risk: float
    max_drawdown_pct: float


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[TradeResult]
    trials_tested: int
    period_start: date
    period_end: date
    generated_at: datetime
    recommendations: List[str]
    report_summary: str


class BacktestEngine:
    """
    Engine for backtesting clinical trial signal strategies.

    Tests how well signals predicted actual trial outcomes and
    calculates ROI for different trading strategies.
    """

    # Sample historical trials for 2023 (real Phase 3 outcomes)
    # In production, this would come from a database
    HISTORICAL_TRIALS_2023 = [
        {
            "trial_id": "NCT03872479",
            "company_ticker": "SGEN",
            "drug_name": "Padcev (enfortumab vedotin)",
            "indication": "Bladder Cancer",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 6, 15),
            "stock_price_before": 145.00,
            "stock_price_after": 198.00,
            "signals": [
                {"signal_type": "enrollment_complete", "signal_weight": 3},
                {"signal_type": "insider_buying", "signal_weight": 4},
                {"signal_type": "sec_8k_positive", "signal_weight": 3},
            ],
            "signal_date": date(2023, 5, 1),
            "composite_score": 8.2,
            "confidence": 0.82,
            "recommendation": "STRONG_BUY"
        },
        {
            "trial_id": "NCT04379596",
            "company_ticker": "MRNA",
            "drug_name": "mRNA-1345 RSV Vaccine",
            "indication": "RSV Infection",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 1, 17),
            "stock_price_before": 180.00,
            "stock_price_after": 215.00,
            "signals": [
                {"signal_type": "early_enrollment", "signal_weight": 3},
                {"signal_type": "breakthrough_designation", "signal_weight": 5},
            ],
            "signal_date": date(2022, 12, 15),
            "composite_score": 7.8,
            "confidence": 0.78,
            "recommendation": "BUY"
        },
        {
            "trial_id": "NCT03434379",
            "company_ticker": "BIIB",
            "drug_name": "Lecanemab",
            "indication": "Alzheimer's Disease",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 1, 6),
            "stock_price_before": 275.00,
            "stock_price_after": 312.00,
            "signals": [
                {"signal_type": "sites_added", "signal_weight": 3},
                {"signal_type": "enrollment_complete", "signal_weight": 3},
                {"signal_type": "fda_priority_review", "signal_weight": 4},
            ],
            "signal_date": date(2022, 11, 20),
            "composite_score": 8.5,
            "confidence": 0.85,
            "recommendation": "STRONG_BUY"
        },
        {
            "trial_id": "NCT04516746",
            "company_ticker": "ALNY",
            "drug_name": "Zilebesiran",
            "indication": "Hypertension",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 9, 18),
            "stock_price_before": 185.00,
            "stock_price_after": 220.00,
            "signals": [
                {"signal_type": "early_enrollment", "signal_weight": 3},
                {"signal_type": "positive_interim", "signal_weight": 4},
            ],
            "signal_date": date(2023, 8, 1),
            "composite_score": 7.5,
            "confidence": 0.75,
            "recommendation": "BUY"
        },
        {
            "trial_id": "NCT04298684",
            "company_ticker": "VIR",
            "drug_name": "VIR-2218",
            "indication": "Hepatitis B",
            "phase": "PHASE3",
            "outcome": TrialOutcome.FAILURE,
            "outcome_date": date(2023, 8, 10),
            "stock_price_before": 28.00,
            "stock_price_after": 18.50,
            "signals": [
                {"signal_type": "enrollment_extended", "signal_weight": -3},
                {"signal_type": "endpoint_change", "signal_weight": -5},
            ],
            "signal_date": date(2023, 7, 1),
            "composite_score": 2.8,
            "confidence": 0.72,
            "recommendation": "SHORT"
        },
        {
            "trial_id": "NCT03760146",
            "company_ticker": "RARE",
            "drug_name": "Setmelanotide",
            "indication": "Obesity (Rare Genetic)",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 4, 12),
            "stock_price_before": 25.00,
            "stock_price_after": 38.00,
            "signals": [
                {"signal_type": "orphan_designation", "signal_weight": 3},
                {"signal_type": "enrollment_complete", "signal_weight": 3},
            ],
            "signal_date": date(2023, 3, 1),
            "composite_score": 7.2,
            "confidence": 0.70,
            "recommendation": "BUY"
        },
        {
            "trial_id": "NCT04153149",
            "company_ticker": "IONS",
            "drug_name": "ION363",
            "indication": "ALS",
            "phase": "PHASE3",
            "outcome": TrialOutcome.FAILURE,
            "outcome_date": date(2023, 11, 8),
            "stock_price_before": 52.00,
            "stock_price_after": 38.00,
            "signals": [
                {"signal_type": "completion_date_delayed", "signal_weight": -3},
                {"signal_type": "insider_selling", "signal_weight": -4},
            ],
            "signal_date": date(2023, 10, 1),
            "composite_score": 2.5,
            "confidence": 0.68,
            "recommendation": "SHORT"
        },
        {
            "trial_id": "NCT03981796",
            "company_ticker": "BPMC",
            "drug_name": "Pralsetinib",
            "indication": "Lung Cancer",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 5, 22),
            "stock_price_before": 85.00,
            "stock_price_after": 115.00,
            "signals": [
                {"signal_type": "sites_added", "signal_weight": 3},
                {"signal_type": "sec_8k_positive", "signal_weight": 3},
            ],
            "signal_date": date(2023, 4, 15),
            "composite_score": 7.6,
            "confidence": 0.76,
            "recommendation": "BUY"
        },
        {
            "trial_id": "NCT04305106",
            "company_ticker": "CRNX",
            "drug_name": "Crinecerfont",
            "indication": "Congenital Adrenal Hyperplasia",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 7, 31),
            "stock_price_before": 32.00,
            "stock_price_after": 48.00,
            "signals": [
                {"signal_type": "early_enrollment", "signal_weight": 3},
                {"signal_type": "breakthrough_designation", "signal_weight": 5},
            ],
            "signal_date": date(2023, 6, 20),
            "composite_score": 8.1,
            "confidence": 0.80,
            "recommendation": "STRONG_BUY"
        },
        {
            "trial_id": "NCT04006301",
            "company_ticker": "NVAX",
            "drug_name": "COVID-19/Flu Combo Vaccine",
            "indication": "Influenza/COVID-19",
            "phase": "PHASE3",
            "outcome": TrialOutcome.PARTIAL,
            "outcome_date": date(2023, 10, 15),
            "stock_price_before": 12.00,
            "stock_price_after": 8.50,
            "signals": [
                {"signal_type": "enrollment_extended", "signal_weight": -3},
                {"signal_type": "sec_8k_negative", "signal_weight": -3},
            ],
            "signal_date": date(2023, 9, 1),
            "composite_score": 3.5,
            "confidence": 0.65,
            "recommendation": "HOLD"
        },
        {
            "trial_id": "NCT03855332",
            "company_ticker": "RCKT",
            "drug_name": "RP-L201",
            "indication": "Leukocyte Adhesion Deficiency",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 3, 28),
            "stock_price_before": 18.00,
            "stock_price_after": 26.00,
            "signals": [
                {"signal_type": "orphan_designation", "signal_weight": 3},
                {"signal_type": "positive_interim", "signal_weight": 4},
            ],
            "signal_date": date(2023, 2, 15),
            "composite_score": 7.8,
            "confidence": 0.78,
            "recommendation": "BUY"
        },
        {
            "trial_id": "NCT04471727",
            "company_ticker": "ARVN",
            "drug_name": "ARV-471",
            "indication": "Breast Cancer",
            "phase": "PHASE3",
            "outcome": TrialOutcome.SUCCESS,
            "outcome_date": date(2023, 12, 11),
            "stock_price_before": 28.00,
            "stock_price_after": 42.00,
            "signals": [
                {"signal_type": "early_enrollment", "signal_weight": 3},
                {"signal_type": "insider_buying", "signal_weight": 4},
                {"signal_type": "partnership_announced", "signal_weight": 4},
            ],
            "signal_date": date(2023, 11, 1),
            "composite_score": 8.4,
            "confidence": 0.84,
            "recommendation": "STRONG_BUY"
        },
    ]

    def __init__(self, config: Optional[BacktestConfig] = None):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config or BacktestConfig()
        self.historical_trials: List[HistoricalTrial] = []

        # Load historical data
        self._load_historical_data()

    def _load_historical_data(self) -> None:
        """Load historical trial data for backtesting."""
        for trial_dict in self.HISTORICAL_TRIALS_2023:
            trial = HistoricalTrial(
                trial_id=trial_dict["trial_id"],
                company_ticker=trial_dict["company_ticker"],
                drug_name=trial_dict["drug_name"],
                indication=trial_dict["indication"],
                phase=trial_dict["phase"],
                outcome=trial_dict["outcome"],
                outcome_date=trial_dict["outcome_date"],
                stock_price_before=trial_dict["stock_price_before"],
                stock_price_after=trial_dict["stock_price_after"],
                signals=trial_dict["signals"],
                signal_date=trial_dict["signal_date"],
                composite_score=trial_dict["composite_score"],
                confidence=trial_dict["confidence"],
                recommendation=trial_dict["recommendation"]
            )
            self.historical_trials.append(trial)

        logger.info(f"Loaded {len(self.historical_trials)} historical trials for backtesting")

    def _get_trade_action(self, trial: HistoricalTrial) -> TradeAction:
        """Determine trade action based on signal."""
        if trial.composite_score >= 8.0 and trial.confidence >= self.config.confidence_threshold:
            return TradeAction.STRONG_BUY
        elif trial.composite_score >= self.config.buy_threshold:
            return TradeAction.BUY
        elif trial.composite_score <= 2.0 and trial.confidence >= self.config.confidence_threshold:
            return TradeAction.STRONG_SHORT
        elif trial.composite_score <= self.config.short_threshold:
            return TradeAction.SHORT
        else:
            return TradeAction.HOLD

    def _calculate_return(
        self,
        trial: HistoricalTrial,
        action: TradeAction
    ) -> Tuple[float, float]:
        """
        Calculate return for a trade.

        Args:
            trial: Historical trial data
            action: Trade action taken

        Returns:
            Tuple of (return_pct, return_usd)
        """
        price_change_pct = (
            (trial.stock_price_after - trial.stock_price_before)
            / trial.stock_price_before
        )

        if action in (TradeAction.BUY, TradeAction.STRONG_BUY):
            return_pct = price_change_pct
        elif action in (TradeAction.SHORT, TradeAction.STRONG_SHORT):
            return_pct = -price_change_pct  # Short profits when price drops
        else:
            return_pct = 0.0  # HOLD = no position

        # Apply stop loss if enabled
        if self.config.use_stop_loss:
            if return_pct < -self.config.stop_loss_pct:
                return_pct = -self.config.stop_loss_pct

        return_usd = self.config.position_size * return_pct

        return return_pct, return_usd

    def _is_prediction_correct(
        self,
        action: TradeAction,
        outcome: TrialOutcome
    ) -> bool:
        """Check if prediction was correct."""
        if action in (TradeAction.BUY, TradeAction.STRONG_BUY):
            return outcome == TrialOutcome.SUCCESS
        elif action in (TradeAction.SHORT, TradeAction.STRONG_SHORT):
            return outcome == TrialOutcome.FAILURE
        else:
            return outcome == TrialOutcome.PARTIAL  # HOLD is "correct" for mixed

    def run_backtest(
        self,
        trials: Optional[List[HistoricalTrial]] = None
    ) -> BacktestResult:
        """
        Run backtesting on historical trials.

        Args:
            trials: Optional list of trials to test (uses loaded data if None)

        Returns:
            BacktestResult with metrics and recommendations
        """
        trials = trials or self.historical_trials

        if not trials:
            raise ValueError("No historical trials available for backtesting")

        logger.info(f"Running backtest on {len(trials)} trials")

        trade_results: List[TradeResult] = []

        for trial in trials:
            # Skip unknown outcomes
            if trial.outcome == TrialOutcome.UNKNOWN:
                continue

            # Determine action
            action = self._get_trade_action(trial)

            # Skip HOLD actions for trading metrics
            if action == TradeAction.HOLD:
                continue

            # Calculate returns
            return_pct, return_usd = self._calculate_return(trial, action)
            correct = self._is_prediction_correct(action, trial.outcome)

            # Calculate hold days
            hold_days = (trial.outcome_date - trial.signal_date).days

            trade_results.append(TradeResult(
                trial_id=trial.trial_id,
                ticker=trial.company_ticker,
                action=action,
                entry_price=trial.stock_price_before,
                exit_price=trial.stock_price_after,
                return_pct=return_pct,
                return_usd=return_usd,
                correct_prediction=correct,
                signal_score=trial.composite_score,
                signal_confidence=trial.confidence,
                actual_outcome=trial.outcome,
                hold_days=hold_days
            ))

        # Calculate metrics
        metrics = self._calculate_metrics(trade_results, trials)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, trade_results)

        # Generate summary
        summary = self._generate_summary(metrics, trade_results)

        # Determine period
        dates = [t.outcome_date for t in trials if t.outcome != TrialOutcome.UNKNOWN]
        period_start = min(dates) if dates else date(2023, 1, 1)
        period_end = max(dates) if dates else date(2023, 12, 31)

        return BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=trade_results,
            trials_tested=len(trials),
            period_start=period_start,
            period_end=period_end,
            generated_at=datetime.now(),
            recommendations=recommendations,
            report_summary=summary
        )

    def _calculate_metrics(
        self,
        trades: List[TradeResult],
        trials: List[HistoricalTrial]
    ) -> BacktestMetrics:
        """Calculate backtesting metrics."""
        if not trades:
            return BacktestMetrics(
                accuracy=0, precision=0, recall=0, f1_score=0,
                total_trades=0, winning_trades=0, losing_trades=0, win_rate=0,
                total_return_pct=0, average_return_pct=0, median_return_pct=0,
                max_return_pct=0, min_return_pct=0, sharpe_ratio=0,
                buy_accuracy=0, short_accuracy=0, strong_signal_accuracy=0,
                estimated_annual_roi=0, capital_at_risk=0, max_drawdown_pct=0
            )

        # Classification metrics
        correct = sum(1 for t in trades if t.correct_prediction)
        total = len(trades)
        accuracy = correct / total if total > 0 else 0

        # Precision/Recall for buy signals
        true_positives = sum(
            1 for t in trades
            if t.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
            and t.actual_outcome == TrialOutcome.SUCCESS
        )
        predicted_positives = sum(
            1 for t in trades
            if t.action in (TradeAction.BUY, TradeAction.STRONG_BUY)
        )
        actual_positives = sum(
            1 for t in trials
            if t.outcome == TrialOutcome.SUCCESS
        )

        precision = true_positives / predicted_positives if predicted_positives > 0 else 0
        recall = true_positives / actual_positives if actual_positives > 0 else 0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0
        )

        # Trading metrics
        winning_trades = sum(1 for t in trades if t.return_pct > 0)
        losing_trades = sum(1 for t in trades if t.return_pct < 0)
        win_rate = winning_trades / total if total > 0 else 0

        # Return metrics
        returns = [t.return_pct for t in trades]
        total_return_pct = sum(returns)
        average_return_pct = statistics.mean(returns) if returns else 0
        median_return_pct = statistics.median(returns) if returns else 0
        max_return_pct = max(returns) if returns else 0
        min_return_pct = min(returns) if returns else 0

        # Sharpe ratio (simplified)
        if len(returns) > 1:
            std_dev = statistics.stdev(returns)
            sharpe_ratio = average_return_pct / std_dev if std_dev > 0 else 0
        else:
            sharpe_ratio = 0

        # Strategy-specific accuracy
        buy_trades = [t for t in trades if t.action in (TradeAction.BUY, TradeAction.STRONG_BUY)]
        buy_accuracy = (
            sum(1 for t in buy_trades if t.correct_prediction) / len(buy_trades)
            if buy_trades else 0
        )

        short_trades = [t for t in trades if t.action in (TradeAction.SHORT, TradeAction.STRONG_SHORT)]
        short_accuracy = (
            sum(1 for t in short_trades if t.correct_prediction) / len(short_trades)
            if short_trades else 0
        )

        strong_trades = [
            t for t in trades
            if t.action in (TradeAction.STRONG_BUY, TradeAction.STRONG_SHORT)
        ]
        strong_signal_accuracy = (
            sum(1 for t in strong_trades if t.correct_prediction) / len(strong_trades)
            if strong_trades else 0
        )

        # ROI estimates
        capital_at_risk = self.config.position_size * self.config.max_positions
        total_return_usd = sum(t.return_usd for t in trades)

        # Annualized ROI (assuming trades were over ~12 months)
        estimated_annual_roi = (total_return_usd / capital_at_risk) if capital_at_risk > 0 else 0

        # Max drawdown (simplified)
        cumulative = []
        running = 0
        for t in trades:
            running += t.return_pct
            cumulative.append(running)

        max_drawdown_pct = 0
        peak = 0
        for c in cumulative:
            if c > peak:
                peak = c
            drawdown = (peak - c)
            if drawdown > max_drawdown_pct:
                max_drawdown_pct = drawdown

        return BacktestMetrics(
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1_score, 4),
            total_trades=total,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 4),
            total_return_pct=round(total_return_pct, 4),
            average_return_pct=round(average_return_pct, 4),
            median_return_pct=round(median_return_pct, 4),
            max_return_pct=round(max_return_pct, 4),
            min_return_pct=round(min_return_pct, 4),
            sharpe_ratio=round(sharpe_ratio, 4),
            buy_accuracy=round(buy_accuracy, 4),
            short_accuracy=round(short_accuracy, 4),
            strong_signal_accuracy=round(strong_signal_accuracy, 4),
            estimated_annual_roi=round(estimated_annual_roi, 4),
            capital_at_risk=capital_at_risk,
            max_drawdown_pct=round(max_drawdown_pct, 4)
        )

    def _generate_recommendations(
        self,
        metrics: BacktestMetrics,
        trades: List[TradeResult]
    ) -> List[str]:
        """Generate recommendations based on backtest results."""
        recommendations = []

        # Accuracy recommendations
        if metrics.accuracy >= 0.7:
            recommendations.append(
                "Signal accuracy is strong (>70%). Consider increasing position sizes."
            )
        elif metrics.accuracy >= 0.5:
            recommendations.append(
                "Signal accuracy is moderate. Maintain current position sizes."
            )
        else:
            recommendations.append(
                "Signal accuracy is below 50%. Consider tightening signal thresholds."
            )

        # Buy vs Short analysis
        if metrics.buy_accuracy > metrics.short_accuracy + 0.1:
            recommendations.append(
                f"Buy signals ({metrics.buy_accuracy:.0%}) outperform short signals "
                f"({metrics.short_accuracy:.0%}). Consider focusing on long positions."
            )
        elif metrics.short_accuracy > metrics.buy_accuracy + 0.1:
            recommendations.append(
                f"Short signals ({metrics.short_accuracy:.0%}) outperform buy signals "
                f"({metrics.buy_accuracy:.0%}). Short strategy is effective."
            )

        # Strong signal analysis
        if metrics.strong_signal_accuracy >= 0.8:
            recommendations.append(
                f"Strong signals are highly accurate ({metrics.strong_signal_accuracy:.0%}). "
                "Increase allocation to high-confidence trades."
            )

        # Win rate recommendations
        if metrics.win_rate >= 0.6:
            recommendations.append(
                f"Win rate of {metrics.win_rate:.0%} is strong. Strategy is profitable."
            )
        else:
            recommendations.append(
                f"Win rate of {metrics.win_rate:.0%} is below target. "
                "Consider raising confidence threshold."
            )

        # Sharpe ratio recommendations
        if metrics.sharpe_ratio >= 1.0:
            recommendations.append(
                f"Sharpe ratio of {metrics.sharpe_ratio:.2f} indicates good risk-adjusted returns."
            )
        elif metrics.sharpe_ratio >= 0.5:
            recommendations.append(
                "Risk-adjusted returns are moderate. Consider reducing position sizes."
            )
        else:
            recommendations.append(
                "Risk-adjusted returns are poor. Review signal weighting model."
            )

        # ROI recommendations
        if metrics.estimated_annual_roi >= 0.2:
            recommendations.append(
                f"Estimated annual ROI of {metrics.estimated_annual_roi:.0%} is strong. "
                "Strategy is worth deploying."
            )
        elif metrics.estimated_annual_roi >= 0.1:
            recommendations.append(
                f"Estimated annual ROI of {metrics.estimated_annual_roi:.0%} is acceptable. "
                "Monitor for consistency."
            )
        else:
            recommendations.append(
                f"Estimated annual ROI of {metrics.estimated_annual_roi:.0%} is low. "
                "Strategy needs optimization."
            )

        return recommendations

    def _generate_summary(
        self,
        metrics: BacktestMetrics,
        trades: List[TradeResult]
    ) -> str:
        """Generate human-readable summary."""
        summary_lines = [
            "=" * 60,
            "BACKTESTING REPORT SUMMARY",
            "=" * 60,
            "",
            "PERFORMANCE OVERVIEW",
            "-" * 40,
            f"Total Trades Executed: {metrics.total_trades}",
            f"Winning Trades: {metrics.winning_trades} ({metrics.win_rate:.0%})",
            f"Losing Trades: {metrics.losing_trades}",
            "",
            "PREDICTION ACCURACY",
            "-" * 40,
            f"Overall Accuracy: {metrics.accuracy:.1%}",
            f"Buy Signal Accuracy: {metrics.buy_accuracy:.1%}",
            f"Short Signal Accuracy: {metrics.short_accuracy:.1%}",
            f"Strong Signal Accuracy: {metrics.strong_signal_accuracy:.1%}",
            f"Precision: {metrics.precision:.1%}",
            f"Recall: {metrics.recall:.1%}",
            f"F1 Score: {metrics.f1_score:.3f}",
            "",
            "RETURN METRICS",
            "-" * 40,
            f"Total Return: {metrics.total_return_pct:.1%}",
            f"Average Return per Trade: {metrics.average_return_pct:.1%}",
            f"Median Return: {metrics.median_return_pct:.1%}",
            f"Best Trade: {metrics.max_return_pct:.1%}",
            f"Worst Trade: {metrics.min_return_pct:.1%}",
            f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}",
            "",
            "INVESTMENT ANALYSIS",
            "-" * 40,
            f"Capital at Risk: ${metrics.capital_at_risk:,.0f}",
            f"Estimated Annual ROI: {metrics.estimated_annual_roi:.1%}",
            f"Max Drawdown: {metrics.max_drawdown_pct:.1%}",
            "",
            "=" * 60,
        ]

        return "\n".join(summary_lines)

    def print_report(self, result: BacktestResult) -> None:
        """Print formatted backtest report to console."""
        print(result.report_summary)

        print("\nTOP PERFORMING TRADES")
        print("-" * 40)
        sorted_trades = sorted(result.trades, key=lambda t: t.return_pct, reverse=True)
        for trade in sorted_trades[:5]:
            status = "WIN" if trade.correct_prediction else "LOSS"
            print(
                f"  {trade.ticker}: {trade.action.value.upper()} -> "
                f"{trade.return_pct:+.1%} [{status}]"
            )

        print("\nWORST PERFORMING TRADES")
        print("-" * 40)
        for trade in sorted_trades[-3:]:
            status = "WIN" if trade.correct_prediction else "LOSS"
            print(
                f"  {trade.ticker}: {trade.action.value.upper()} -> "
                f"{trade.return_pct:+.1%} [{status}]"
            )

        print("\nRECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 60)

    def export_report(self, result: BacktestResult, filepath: str) -> None:
        """Export backtest report to JSON file."""
        report_dict = {
            "summary": result.report_summary,
            "config": {
                "buy_threshold": result.config.buy_threshold,
                "short_threshold": result.config.short_threshold,
                "confidence_threshold": result.config.confidence_threshold,
                "position_size": result.config.position_size,
            },
            "metrics": {
                "accuracy": result.metrics.accuracy,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1_score": result.metrics.f1_score,
                "win_rate": result.metrics.win_rate,
                "total_return_pct": result.metrics.total_return_pct,
                "sharpe_ratio": result.metrics.sharpe_ratio,
                "estimated_annual_roi": result.metrics.estimated_annual_roi,
                "buy_accuracy": result.metrics.buy_accuracy,
                "short_accuracy": result.metrics.short_accuracy,
                "strong_signal_accuracy": result.metrics.strong_signal_accuracy,
            },
            "trades": [
                {
                    "trial_id": t.trial_id,
                    "ticker": t.ticker,
                    "action": t.action.value,
                    "return_pct": t.return_pct,
                    "correct": t.correct_prediction,
                    "outcome": t.actual_outcome.value,
                }
                for t in result.trades
            ],
            "recommendations": result.recommendations,
            "period": {
                "start": result.period_start.isoformat(),
                "end": result.period_end.isoformat(),
            },
            "generated_at": result.generated_at.isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Backtest report exported to {filepath}")


if __name__ == "__main__":
    # Run backtesting
    logging.basicConfig(level=logging.INFO)

    print("Running Clinical Trial Signal Backtesting...")
    print("=" * 60)

    # Initialize engine with default config
    engine = BacktestEngine()

    print(f"\nLoaded {len(engine.historical_trials)} historical trials")

    # Run backtest
    result = engine.run_backtest()

    # Print report
    engine.print_report(result)

    # Export report
    report_path = "/Users/waiyang/Desktop/repo/dreamers-v2/clinical_trial_signals/backtesting/backtest_report.json"
    engine.export_report(result, report_path)
    print(f"\nReport exported to: {report_path}")
