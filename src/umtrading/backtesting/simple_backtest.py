"""
Simple Backtesting Engine
Implements backtesting with transaction costs for passive portfolio strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BacktestResults:
    """Container for backtest results"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float
    alpha: float
    beta: float
    win_rate: float
    avg_win: float
    avg_loss: float
    total_trades: int
    total_commission: float
    total_slippage: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.DataFrame
    trades: List[Dict]
    benchmark_return: Optional[float] = None
    benchmark_sharpe: Optional[float] = None
    excess_return: Optional[float] = None


class SimpleBacktest:
    """
    Simple backtesting engine for portfolio strategies
    """

    def __init__(self,
                 start_date: str,
                 end_date: str,
                 initial_capital: float = 100000.0,
                 commission: float = 1.0,
                 slippage: float = 0.001,
                 risk_free_rate: float = 0.02):
        """
        Initialize backtesting engine

        Args:
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            initial_capital: Initial capital
            commission: Commission per trade in dollars
            slippage: Slippage as percentage (0.001 = 0.1%)
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.risk_free_rate = risk_free_rate

        # Data storage
        self.price_data = {}
        self.benchmark_data = None

    def fetch_data(self, symbols: List[str], benchmark: str = 'SPY') -> bool:
        """
        Fetch historical price data for symbols and benchmark

        Args:
            symbols: List of stock symbols
            benchmark: Benchmark symbol (default: SPY)

        Returns:
            True if data fetched successfully
        """
        print(f"Fetching data for {len(symbols)} symbols from {self.start_date.date()} to {self.end_date.date()}")

        try:
            # Fetch data for all symbols
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=self.start_date, end=self.end_date)

                if data.empty:
                    print(f"Warning: No data for {symbol}")
                    continue

                self.price_data[symbol] = data[['Open', 'High', 'Low', 'Close', 'Volume']]
                print(f"  Fetched {symbol}: {len(data)} days")

            # Fetch benchmark data
            benchmark_ticker = yf.Ticker(benchmark)
            self.benchmark_data = benchmark_ticker.history(start=self.start_date, end=self.end_date)
            print(f"  Fetched {benchmark}: {len(self.benchmark_data)} days")

            return True

        except Exception as e:
            print(f"Error fetching data: {e}")
            return False

    def run_backtest(self, strategy) -> BacktestResults:
        """
        Run backtest for a given strategy

        Args:
            strategy: Strategy object with required methods

        Returns:
            BacktestResults object with performance metrics
        """
        print(f"\nRunning backtest for {strategy.__class__.__name__}")
        print("=" * 50)

        # Reset strategy
        strategy.reset()

        # Prepare data structures
        equity_curve = []
        daily_returns = []
        all_trades = []
        position_history = {symbol: [] for symbol in strategy.symbols}
        dates = []

        # Get all trading dates
        all_dates = set()
        for symbol_data in self.price_data.values():
            all_dates.update(symbol_data.index)
        all_dates = sorted(list(all_dates))

        # Variables for tracking
        last_portfolio_value = self.initial_capital
        total_commission = 0.0
        total_slippage = 0.0

        # Run backtest day by day
        for date in all_dates:
            # Get current prices
            current_prices = {}
            for symbol in strategy.symbols:
                if symbol in self.price_data:
                    symbol_data = self.price_data[symbol]
                    if date in symbol_data.index:
                        current_prices[symbol] = symbol_data.loc[date, 'Close']

            # Skip if we don't have prices for all symbols
            if len(current_prices) < len(strategy.symbols):
                continue

            # Check if we should rebalance
            if strategy.should_rebalance(date, current_prices):
                # Calculate rebalancing trades
                trades = strategy.calculate_rebalance_trades(
                    date, current_prices, self.commission, self.slippage
                )

                if trades:
                    # Execute trades
                    execution_report = strategy.execute_trades(
                        trades, current_prices, date, self.commission, self.slippage
                    )

                    # Track trades
                    all_trades.append(execution_report)
                    total_commission += execution_report['costs']['commission']
                    total_slippage += execution_report['costs']['slippage']

                    print(f"  Rebalanced on {date.date()}: {len(trades)} trades, "
                          f"costs=${execution_report['costs']['total']:.2f}")

            # Calculate portfolio value
            portfolio_value = strategy.get_portfolio_value(current_prices)

            # Track daily values
            dates.append(date)
            equity_curve.append(portfolio_value)

            # Calculate daily return
            if last_portfolio_value > 0:
                daily_return = (portfolio_value - last_portfolio_value) / last_portfolio_value
            else:
                daily_return = 0.0
            daily_returns.append(daily_return)
            last_portfolio_value = portfolio_value

            # Track positions
            current_positions = strategy.get_positions()
            for symbol in strategy.symbols:
                position_history[symbol].append(current_positions.get(symbol, 0))

        # Create DataFrames
        equity_curve = pd.Series(equity_curve, index=dates, name='Portfolio Value')
        daily_returns = pd.Series(daily_returns, index=dates, name='Daily Returns')
        positions_df = pd.DataFrame(position_history, index=dates)

        # Calculate performance metrics
        results = self._calculate_metrics(
            equity_curve, daily_returns, positions_df, all_trades,
            total_commission, total_slippage
        )

        # Calculate benchmark performance if available
        if self.benchmark_data is not None and len(self.benchmark_data) > 0:
            benchmark_results = self._calculate_benchmark_metrics()
            results.benchmark_return = benchmark_results['total_return']
            results.benchmark_sharpe = benchmark_results['sharpe_ratio']
            results.excess_return = results.total_return - benchmark_results['total_return']
            results.alpha = results.annualized_return - benchmark_results['annualized_return']

        return results

    def _calculate_metrics(self,
                          equity_curve: pd.Series,
                          daily_returns: pd.Series,
                          positions: pd.DataFrame,
                          trades: List[Dict],
                          total_commission: float,
                          total_slippage: float) -> BacktestResults:
        """Calculate performance metrics from backtest data"""

        # Basic returns
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

        # Risk metrics
        returns_clean = daily_returns[daily_returns != 0]  # Remove zero returns
        if len(returns_clean) > 0:
            volatility = returns_clean.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0

            # Sortino ratio (downside deviation)
            downside_returns = returns_clean[returns_clean < 0]
            if len(downside_returns) > 0:
                downside_std = downside_returns.std() * np.sqrt(252)
                sortino_ratio = (annualized_return - self.risk_free_rate) / downside_std if downside_std > 0 else 0
            else:
                sortino_ratio = sharpe_ratio * 2  # No losses, excellent!
        else:
            volatility = 0
            sharpe_ratio = 0
            sortino_ratio = 0

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win/loss statistics
        positive_returns = returns_clean[returns_clean > 0]
        negative_returns = returns_clean[returns_clean < 0]

        win_rate = len(positive_returns) / len(returns_clean) if len(returns_clean) > 0 else 0
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0
        avg_loss = negative_returns.mean() if len(negative_returns) > 0 else 0

        # Beta calculation (if we have benchmark)
        beta = 1.0  # Default beta

        return BacktestResults(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            alpha=0.0,  # Will be calculated vs benchmark
            beta=beta,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            total_trades=len(trades),
            total_commission=total_commission,
            total_slippage=total_slippage,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            positions=positions,
            trades=trades
        )

    def _calculate_benchmark_metrics(self) -> Dict[str, float]:
        """Calculate benchmark (SPY) performance metrics"""
        if self.benchmark_data is None or len(self.benchmark_data) == 0:
            return {'total_return': 0, 'annualized_return': 0, 'sharpe_ratio': 0}

        # Calculate returns
        initial_price = self.benchmark_data['Close'].iloc[0]
        final_price = self.benchmark_data['Close'].iloc[-1]
        total_return = (final_price - initial_price) / initial_price

        # Annualized return
        n_years = (self.benchmark_data.index[-1] - self.benchmark_data.index[0]).days / 365.25
        annualized_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0

        # Daily returns
        daily_returns = self.benchmark_data['Close'].pct_change().dropna()

        # Sharpe ratio
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility
        }

    def print_results(self, results: BacktestResults, strategy_name: str = "Strategy"):
        """
        Print formatted backtest results

        Args:
            results: BacktestResults object
            strategy_name: Name of the strategy
        """
        print(f"\n{'='*60}")
        print(f"  BACKTEST RESULTS: {strategy_name}")
        print(f"{'='*60}")

        print(f"\n📊 RETURNS:")
        print(f"  Total Return:        {results.total_return*100:>8.2f}%")
        print(f"  Annualized Return:   {results.annualized_return*100:>8.2f}%")
        print(f"  Volatility:          {results.volatility*100:>8.2f}%")

        print(f"\n📈 RISK-ADJUSTED RETURNS:")
        print(f"  Sharpe Ratio:        {results.sharpe_ratio:>8.2f}")
        print(f"  Sortino Ratio:       {results.sortino_ratio:>8.2f}")
        print(f"  Calmar Ratio:        {results.calmar_ratio:>8.2f}")

        print(f"\n📉 RISK METRICS:")
        print(f"  Max Drawdown:        {results.max_drawdown*100:>8.2f}%")
        print(f"  Win Rate:            {results.win_rate*100:>8.2f}%")
        print(f"  Avg Win:             {results.avg_win*100:>8.2f}%")
        print(f"  Avg Loss:            {results.avg_loss*100:>8.2f}%")

        print(f"\n💰 TRADING COSTS:")
        print(f"  Total Trades:        {results.total_trades:>8}")
        print(f"  Total Commission:    ${results.total_commission:>8.2f}")
        print(f"  Total Slippage:      ${results.total_slippage:>8.2f}")
        print(f"  Total Costs:         ${results.total_commission + results.total_slippage:>8.2f}")

        if results.benchmark_return is not None:
            print(f"\n🎯 VS BENCHMARK (SPY):")
            print(f"  Benchmark Return:    {results.benchmark_return*100:>8.2f}%")
            print(f"  Benchmark Sharpe:    {results.benchmark_sharpe:>8.2f}")
            print(f"  Excess Return:       {results.excess_return*100:>8.2f}%")
            print(f"  Alpha:               {results.alpha*100:>8.2f}%")

        print(f"\n{'='*60}\n")