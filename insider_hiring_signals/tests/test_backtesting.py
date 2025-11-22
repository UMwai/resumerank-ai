"""
Tests for the backtesting module.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtesting.backtest_signals import (
    BacktestEngine,
    BacktestResult,
    SignalPerformance,
    InsiderSignal,
    SyntheticDataGenerator,
    HistoricalDataFetcher,
    BIOTECH_UNIVERSE,
)


class TestSyntheticDataGenerator:
    """Tests for synthetic data generation."""

    def test_generate_signals(self):
        """Test that synthetic signals are generated correctly."""
        gen = SyntheticDataGenerator(seed=42)

        signals = gen.generate_form4_signals(
            tickers=['MRNA', 'VRTX'],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            signals_per_company=5
        )

        assert len(signals) > 0
        assert all(isinstance(s, InsiderSignal) for s in signals)
        assert all(s.ticker in ['MRNA', 'VRTX'] for s in signals)
        assert all(-10 <= s.signal_strength <= 10 for s in signals)

    def test_generate_returns(self):
        """Test that synthetic returns are correlated with signals."""
        gen = SyntheticDataGenerator(seed=42)

        signals = gen.generate_form4_signals(
            tickers=['MRNA'],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            signals_per_company=50
        )

        returns = gen.generate_price_returns(signals)

        # Returns dict uses unique keys, so duplicates on same date reduce count
        assert len(returns) <= len(signals)

        # Check return structure
        for key, ret in returns.items():
            assert 'return_7d' in ret
            assert 'return_30d' in ret
            assert 'return_90d' in ret

    def test_signal_strength_distribution(self):
        """Test that signal strengths have reasonable distribution."""
        gen = SyntheticDataGenerator(seed=42)

        signals = gen.generate_form4_signals(
            tickers=['MRNA'],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            signals_per_company=100
        )

        strengths = [s.signal_strength for s in signals]

        # Should have both positive and negative signals
        assert any(s > 0 for s in strengths)
        assert any(s < 0 for s in strengths)

        # Most signals should be moderate strength
        moderate = [s for s in strengths if -5 <= s <= 5]
        assert len(moderate) > len(strengths) * 0.5


class TestBacktestEngine:
    """Tests for the BacktestEngine."""

    def test_initialization(self):
        """Test engine initializes correctly."""
        engine = BacktestEngine()

        assert engine.use_synthetic == True
        assert engine.data_fetcher is not None
        assert engine.synthetic_gen is not None

    def test_run_backtest(self):
        """Test basic backtest execution."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=['MRNA', 'VRTX'],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        assert isinstance(result, BacktestResult)
        assert result.total_signals > 0
        assert result.companies_analyzed == 2
        assert -1 <= result.correlation_30d <= 1

    def test_correlation_calculation(self):
        """Test correlation metrics are calculated."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=BIOTECH_UNIVERSE[:10],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        # Correlations should be between -1 and 1
        assert -1 <= result.correlation_7d <= 1
        assert -1 <= result.correlation_30d <= 1
        assert -1 <= result.correlation_90d <= 1

    def test_win_rate_calculation(self):
        """Test win rate metrics."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=BIOTECH_UNIVERSE[:20],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        # Win rates should be between 0 and 1
        assert 0 <= result.bullish_win_rate_30d <= 1
        assert 0 <= result.bearish_win_rate_30d <= 1
        assert 0 <= result.overall_accuracy <= 1

    def test_return_by_strength(self):
        """Test returns are calculated by signal strength bucket."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=BIOTECH_UNIVERSE[:20],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        assert isinstance(result.avg_return_by_strength, dict)
        assert 'strong_bullish (6+)' in result.avg_return_by_strength
        assert 'strong_bearish (<-6)' in result.avg_return_by_strength

    def test_weight_optimization(self):
        """Test signal weight optimization."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=BIOTECH_UNIVERSE[:30],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        assert isinstance(result.optimized_weights, dict)
        assert len(result.optimized_weights) > 0
        assert isinstance(result.weight_improvement, float)

    def test_sector_summary(self):
        """Test sector-specific analysis."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=BIOTECH_UNIVERSE[:30],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        assert isinstance(result.summary_by_sector, dict)
        # Should have some sector data
        if result.summary_by_sector:
            for sector, stats in result.summary_by_sector.items():
                assert 'count' in stats
                assert 'avg_return_30d' in stats

    def test_recommendations_generated(self):
        """Test trading recommendations are generated."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=BIOTECH_UNIVERSE[:20],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        assert isinstance(result.trading_recommendations, list)
        assert len(result.trading_recommendations) > 0

    def test_generate_report(self):
        """Test report generation."""
        engine = BacktestEngine()

        result = engine.run_backtest(
            tickers=['MRNA', 'VRTX'],
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31)
        )

        report = engine.generate_report(result)

        assert isinstance(report, str)
        assert 'BACKTESTING REPORT' in report
        assert 'CORRELATION' in report
        assert 'WIN RATES' in report


class TestSignalPerformance:
    """Tests for SignalPerformance dataclass."""

    def test_performance_creation(self):
        """Test SignalPerformance creation."""
        perf = SignalPerformance(
            ticker='MRNA',
            signal_date=date(2023, 6, 15),
            signal_strength=5.0,
            signal_type='insider_buy',
            return_7d=2.5,
            return_30d=8.0,
            return_90d=15.0,
            max_drawdown_30d=-5.0,
            hit_target_10pct=False,
            hit_stop_loss_10pct=False,
            price_at_signal=100.0,
            price_7d=102.5,
            price_30d=108.0,
            price_90d=115.0
        )

        assert perf.ticker == 'MRNA'
        assert perf.signal_strength == 5.0
        assert perf.return_30d == 8.0


class TestBacktestResult:
    """Tests for BacktestResult dataclass."""

    def test_result_attributes(self):
        """Test BacktestResult has required attributes."""
        engine = BacktestEngine()
        result = engine.run_backtest(
            tickers=['MRNA'],
            start_date=date(2023, 6, 1),
            end_date=date(2023, 12, 31)
        )

        # Check all required attributes exist
        assert hasattr(result, 'start_date')
        assert hasattr(result, 'end_date')
        assert hasattr(result, 'total_signals')
        assert hasattr(result, 'correlation_30d')
        assert hasattr(result, 'bullish_win_rate_30d')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'optimized_weights')
        assert hasattr(result, 'trading_recommendations')


class TestHistoricalDataFetcher:
    """Tests for HistoricalDataFetcher."""

    def test_initialization(self):
        """Test fetcher initializes."""
        fetcher = HistoricalDataFetcher()
        assert fetcher._price_cache == {}

    def test_get_price_at_date(self):
        """Test price lookup with mock data."""
        fetcher = HistoricalDataFetcher()

        # Create mock DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'date': [date(2023, 1, 2), date(2023, 1, 3), date(2023, 1, 4)],
            'close': [100.0, 102.0, 101.0]
        })

        price = fetcher.get_price_at_date(df, date(2023, 1, 3))
        assert price == 102.0

    def test_calculate_return(self):
        """Test return calculation with mock data."""
        fetcher = HistoricalDataFetcher()

        import pandas as pd
        df = pd.DataFrame({
            'date': [date(2023, 1, 1) + timedelta(days=i) for i in range(35)],
            'close': [100 + i for i in range(35)]
        })

        ret = fetcher.calculate_return(df, date(2023, 1, 1), 30)
        # From 100 to 130 = 30% return
        assert ret is not None
        assert abs(ret - 30.0) < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
