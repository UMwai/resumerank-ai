"""
Tests for the Enhanced Backtesting Framework.
"""
import pytest
import json
import tempfile
from datetime import date, datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backtesting.backtest_engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    BacktestMetrics,
    HistoricalTrial,
    TradeResult,
    TrialOutcome,
    TradeAction,
)


class TestBacktestConfig:
    """Tests for BacktestConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.buy_threshold == 7.0
        assert config.short_threshold == 3.0
        assert config.confidence_threshold == 0.7
        assert config.position_size == 10000.0
        assert config.max_positions == 5

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            buy_threshold=8.0,
            short_threshold=2.0,
            position_size=20000.0
        )

        assert config.buy_threshold == 8.0
        assert config.short_threshold == 2.0
        assert config.position_size == 20000.0


class TestHistoricalTrial:
    """Tests for HistoricalTrial dataclass."""

    def test_trial_creation(self):
        """Test creating a historical trial."""
        trial = HistoricalTrial(
            trial_id="NCT12345678",
            company_ticker="BTCH",
            drug_name="TestDrug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.SUCCESS,
            outcome_date=date(2023, 6, 15),
            stock_price_before=100.0,
            stock_price_after=150.0,
            signals=[{"signal_type": "insider_buying", "signal_weight": 4}],
            signal_date=date(2023, 5, 1),
            composite_score=8.5,
            confidence=0.85,
            recommendation="STRONG_BUY"
        )

        assert trial.trial_id == "NCT12345678"
        assert trial.outcome == TrialOutcome.SUCCESS
        assert trial.stock_price_after > trial.stock_price_before


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine()

    def test_engine_initialization(self):
        """Test engine initializes with historical data."""
        assert len(self.engine.historical_trials) > 0

    def test_engine_with_custom_config(self):
        """Test engine with custom configuration."""
        config = BacktestConfig(buy_threshold=8.0)
        engine = BacktestEngine(config=config)

        assert engine.config.buy_threshold == 8.0

    def test_get_trade_action_strong_buy(self):
        """Test trade action for strong buy signal."""
        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.SUCCESS,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=150,
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=8.5,
            confidence=0.85,
            recommendation="STRONG_BUY"
        )

        action = self.engine._get_trade_action(trial)
        assert action == TradeAction.STRONG_BUY

    def test_get_trade_action_short(self):
        """Test trade action for short signal."""
        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.FAILURE,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=50,
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=2.5,
            confidence=0.70,
            recommendation="SHORT"
        )

        action = self.engine._get_trade_action(trial)
        assert action == TradeAction.SHORT

    def test_get_trade_action_hold(self):
        """Test trade action for hold signal."""
        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.PARTIAL,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=100,
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=5.0,
            confidence=0.60,
            recommendation="HOLD"
        )

        action = self.engine._get_trade_action(trial)
        assert action == TradeAction.HOLD

    def test_calculate_return_buy(self):
        """Test return calculation for buy action."""
        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.SUCCESS,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=150,  # 50% increase
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=8.0,
            confidence=0.80,
            recommendation="BUY"
        )

        return_pct, return_usd = self.engine._calculate_return(trial, TradeAction.BUY)

        assert return_pct == 0.5  # 50%
        assert return_usd == 5000.0  # 50% of $10,000

    def test_calculate_return_short(self):
        """Test return calculation for short action."""
        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.FAILURE,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=70,  # 30% decrease
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=2.5,
            confidence=0.75,
            recommendation="SHORT"
        )

        return_pct, return_usd = self.engine._calculate_return(trial, TradeAction.SHORT)

        assert return_pct == 0.3  # 30% profit from short
        assert return_usd == 3000.0

    def test_is_prediction_correct_buy_success(self):
        """Test prediction correctness for successful buy."""
        correct = self.engine._is_prediction_correct(
            TradeAction.BUY,
            TrialOutcome.SUCCESS
        )
        assert correct is True

    def test_is_prediction_correct_buy_failure(self):
        """Test prediction correctness for failed buy."""
        correct = self.engine._is_prediction_correct(
            TradeAction.BUY,
            TrialOutcome.FAILURE
        )
        assert correct is False

    def test_is_prediction_correct_short_failure(self):
        """Test prediction correctness for successful short."""
        correct = self.engine._is_prediction_correct(
            TradeAction.SHORT,
            TrialOutcome.FAILURE
        )
        assert correct is True

    def test_run_backtest(self):
        """Test running complete backtest."""
        result = self.engine.run_backtest()

        assert isinstance(result, BacktestResult)
        assert result.trials_tested == len(self.engine.historical_trials)
        assert result.metrics is not None
        assert len(result.trades) > 0
        assert len(result.recommendations) > 0

    def test_backtest_metrics(self):
        """Test backtest metrics calculation."""
        result = self.engine.run_backtest()
        metrics = result.metrics

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1
        assert metrics.total_trades > 0
        assert metrics.winning_trades + metrics.losing_trades == metrics.total_trades

    def test_backtest_period(self):
        """Test backtest period is set correctly."""
        result = self.engine.run_backtest()

        assert result.period_start is not None
        assert result.period_end is not None
        assert result.period_start <= result.period_end

    def test_backtest_recommendations(self):
        """Test backtest generates recommendations."""
        result = self.engine.run_backtest()

        assert len(result.recommendations) > 0
        for rec in result.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_backtest_summary(self):
        """Test backtest generates summary."""
        result = self.engine.run_backtest()

        assert len(result.report_summary) > 0
        assert "PERFORMANCE" in result.report_summary or "SUMMARY" in result.report_summary


class TestBacktestMetrics:
    """Tests for BacktestMetrics calculations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine()

    def test_calculate_metrics_empty(self):
        """Test metrics calculation with empty trades."""
        metrics = self.engine._calculate_metrics([], [])

        assert metrics.accuracy == 0
        assert metrics.total_trades == 0
        assert metrics.win_rate == 0

    def test_win_rate_calculation(self):
        """Test win rate is calculated correctly."""
        result = self.engine.run_backtest()

        expected_win_rate = (
            result.metrics.winning_trades / result.metrics.total_trades
            if result.metrics.total_trades > 0 else 0
        )

        assert abs(result.metrics.win_rate - expected_win_rate) < 0.01


class TestBacktestReport:
    """Tests for backtest report generation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = BacktestEngine()
        self.result = self.engine.run_backtest()

    def test_print_report(self, capsys):
        """Test printing report to console."""
        self.engine.print_report(self.result)
        captured = capsys.readouterr()

        assert "PERFORMANCE" in captured.out or "SUMMARY" in captured.out
        assert "RECOMMENDATIONS" in captured.out

    def test_export_report_json(self):
        """Test exporting report to JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        self.engine.export_report(self.result, filepath)

        with open(filepath, 'r') as f:
            data = json.load(f)

        assert "metrics" in data
        assert "recommendations" in data
        assert "trades" in data
        assert data["metrics"]["accuracy"] == self.result.metrics.accuracy


class TestTrialOutcomes:
    """Tests for trial outcome handling."""

    def test_outcome_enum_values(self):
        """Test all outcome enum values exist."""
        assert TrialOutcome.SUCCESS.value == "success"
        assert TrialOutcome.FAILURE.value == "failure"
        assert TrialOutcome.PARTIAL.value == "partial"
        assert TrialOutcome.TERMINATED.value == "terminated"
        assert TrialOutcome.UNKNOWN.value == "unknown"


class TestTradeActions:
    """Tests for trade action handling."""

    def test_trade_action_enum_values(self):
        """Test all trade action enum values exist."""
        assert TradeAction.STRONG_BUY.value == "strong_buy"
        assert TradeAction.BUY.value == "buy"
        assert TradeAction.HOLD.value == "hold"
        assert TradeAction.SHORT.value == "short"
        assert TradeAction.STRONG_SHORT.value == "strong_short"


class TestTradeResult:
    """Tests for TradeResult dataclass."""

    def test_trade_result_creation(self):
        """Test creating a trade result."""
        result = TradeResult(
            trial_id="NCT12345678",
            ticker="BTCH",
            action=TradeAction.BUY,
            entry_price=100.0,
            exit_price=150.0,
            return_pct=0.5,
            return_usd=5000.0,
            correct_prediction=True,
            signal_score=8.5,
            signal_confidence=0.85,
            actual_outcome=TrialOutcome.SUCCESS,
            hold_days=45
        )

        assert result.trial_id == "NCT12345678"
        assert result.return_pct == 0.5
        assert result.correct_prediction is True


class TestStopLoss:
    """Tests for stop loss functionality."""

    def test_stop_loss_applied(self):
        """Test stop loss limits losses."""
        config = BacktestConfig(
            use_stop_loss=True,
            stop_loss_pct=0.15
        )
        engine = BacktestEngine(config=config)

        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.FAILURE,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=50,  # 50% loss
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=8.0,
            confidence=0.80,
            recommendation="BUY"
        )

        return_pct, _ = engine._calculate_return(trial, TradeAction.BUY)

        # Should be limited to -15%
        assert return_pct == -0.15

    def test_stop_loss_not_triggered(self):
        """Test stop loss not triggered for small losses."""
        config = BacktestConfig(
            use_stop_loss=True,
            stop_loss_pct=0.15
        )
        engine = BacktestEngine(config=config)

        trial = HistoricalTrial(
            trial_id="NCT001",
            company_ticker="TEST",
            drug_name="Drug",
            indication="Cancer",
            phase="PHASE3",
            outcome=TrialOutcome.PARTIAL,
            outcome_date=date(2023, 6, 1),
            stock_price_before=100,
            stock_price_after=95,  # 5% loss
            signals=[],
            signal_date=date(2023, 5, 1),
            composite_score=8.0,
            confidence=0.80,
            recommendation="BUY"
        )

        return_pct, _ = engine._calculate_return(trial, TradeAction.BUY)

        # Should be actual loss, not stop loss
        assert return_pct == -0.05
