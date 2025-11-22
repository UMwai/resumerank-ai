"""
Backtesting module for validating insider/hiring signals against historical performance.
"""

from .backtest_signals import BacktestEngine, BacktestResult, SignalPerformance

__all__ = ['BacktestEngine', 'BacktestResult', 'SignalPerformance']
