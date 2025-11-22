"""
Run Comprehensive Backtest for Passive Portfolio Strategy
Tests passive portfolio vs SPY benchmark from 2020-2024
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yfinance as yf

from src.umtrading.strategies.passive_portfolio import PassivePortfolio
from src.umtrading.backtesting.simple_backtest import SimpleBacktest, BacktestResults

import warnings
warnings.filterwarnings('ignore')


def run_spy_benchmark_backtest(start_date: str, end_date: str, initial_capital: float = 100000.0) -> BacktestResults:
    """
    Run a simple buy-and-hold backtest for SPY benchmark

    Args:
        start_date: Start date for backtest
        end_date: End date for backtest
        initial_capital: Initial capital

    Returns:
        BacktestResults for SPY buy-and-hold
    """
    print("\nRunning SPY Buy-and-Hold Benchmark")
    print("=" * 50)

    # Fetch SPY data
    spy = yf.Ticker('SPY')
    spy_data = spy.history(start=start_date, end=end_date)

    if spy_data.empty:
        print("Error: Could not fetch SPY data")
        return None

    # Calculate buy and hold returns
    initial_price = spy_data['Close'].iloc[0]
    shares = initial_capital / initial_price  # Buy as many shares as possible

    # Create equity curve
    equity_curve = spy_data['Close'] * shares
    equity_curve.name = 'Portfolio Value'

    # Calculate daily returns
    daily_returns = spy_data['Close'].pct_change()
    daily_returns.iloc[0] = 0
    daily_returns.name = 'Daily Returns'

    # Calculate metrics
    total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
    n_years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    annualized_return = (1 + total_return) ** (1/n_years) - 1

    # Risk metrics
    returns_clean = daily_returns[daily_returns != 0]
    volatility = returns_clean.std() * np.sqrt(252)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0

    # Sortino ratio
    downside_returns = returns_clean[returns_clean < 0]
    if len(downside_returns) > 0:
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_std
    else:
        sortino_ratio = sharpe_ratio * 2

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

    return BacktestResults(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        volatility=volatility,
        alpha=0.0,
        beta=1.0,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        total_trades=1,  # One initial buy
        total_commission=1.0,  # One trade commission
        total_slippage=initial_price * shares * 0.001,  # 0.1% slippage on initial buy
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        positions=pd.DataFrame({'SPY': [shares] * len(equity_curve)}, index=equity_curve.index),
        trades=[{
            'date': equity_curve.index[0],
            'trades': {'SPY': {'size': shares, 'price': initial_price}},
            'costs': {'commission': 1.0, 'slippage': initial_price * shares * 0.001}
        }]
    )


def create_performance_report(passive_results: BacktestResults, spy_results: BacktestResults):
    """
    Create a comprehensive performance report comparing strategies

    Args:
        passive_results: Results from passive portfolio strategy
        spy_results: Results from SPY benchmark
    """
    print("\n" + "=" * 80)
    print("                    COMPREHENSIVE PERFORMANCE REPORT")
    print("=" * 80)

    # Comparison table
    print("\n📊 STRATEGY COMPARISON")
    print("-" * 60)
    print(f"{'Metric':<25} {'Passive Portfolio':>18} {'SPY B&H':>15}")
    print("-" * 60)

    metrics = [
        ('Total Return (%)', passive_results.total_return * 100, spy_results.total_return * 100),
        ('Annualized Return (%)', passive_results.annualized_return * 100, spy_results.annualized_return * 100),
        ('Volatility (%)', passive_results.volatility * 100, spy_results.volatility * 100),
        ('Sharpe Ratio', passive_results.sharpe_ratio, spy_results.sharpe_ratio),
        ('Sortino Ratio', passive_results.sortino_ratio, spy_results.sortino_ratio),
        ('Max Drawdown (%)', passive_results.max_drawdown * 100, spy_results.max_drawdown * 100),
        ('Calmar Ratio', passive_results.calmar_ratio, spy_results.calmar_ratio),
        ('Win Rate (%)', passive_results.win_rate * 100, spy_results.win_rate * 100),
    ]

    for metric_name, passive_val, spy_val in metrics:
        if '%' in metric_name:
            print(f"{metric_name:<25} {passive_val:>17.2f}% {spy_val:>14.2f}%")
        else:
            print(f"{metric_name:<25} {passive_val:>18.2f} {spy_val:>15.2f}")

    # Calculate alpha
    alpha = passive_results.total_return - spy_results.total_return
    print("-" * 60)
    print(f"{'ALPHA (Excess Return):':<25} {alpha * 100:>17.2f}%")
    print("-" * 60)

    # Performance validation
    print("\n✅ PERFORMANCE VALIDATION")
    print("-" * 60)

    validations = [
        ("Sharpe Ratio > 1.0", passive_results.sharpe_ratio > 1.0),
        ("Max Drawdown < 15%", abs(passive_results.max_drawdown) < 0.15),
        ("Positive Alpha vs SPY", alpha > 0),
        ("Alpha in 20-30% range", 0.20 <= alpha <= 0.30),
    ]

    for check, passed in validations:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {check:<35} {status}")

    # Trading statistics
    print("\n📈 TRADING STATISTICS")
    print("-" * 60)
    print(f"  Total Rebalances:      {passive_results.total_trades}")
    print(f"  Avg Trades/Year:       {passive_results.total_trades / 5:.1f}")
    print(f"  Total Commission:      ${passive_results.total_commission:.2f}")
    print(f"  Total Slippage:        ${passive_results.total_slippage:.2f}")
    print(f"  Total Trading Costs:   ${passive_results.total_commission + passive_results.total_slippage:.2f}")
    print(f"  Costs as % of Return:  {(passive_results.total_commission + passive_results.total_slippage) / (100000 * passive_results.total_return) * 100:.2f}%")

    # Risk-adjusted performance
    print("\n🎯 RISK-ADJUSTED OUTPERFORMANCE")
    print("-" * 60)
    sharpe_improvement = (passive_results.sharpe_ratio - spy_results.sharpe_ratio) / spy_results.sharpe_ratio * 100
    sortino_improvement = (passive_results.sortino_ratio - spy_results.sortino_ratio) / spy_results.sortino_ratio * 100
    dd_improvement = (spy_results.max_drawdown - passive_results.max_drawdown) / abs(spy_results.max_drawdown) * 100

    print(f"  Sharpe Ratio Improvement:  {sharpe_improvement:>8.1f}%")
    print(f"  Sortino Ratio Improvement: {sortino_improvement:>8.1f}%")
    print(f"  Drawdown Reduction:         {dd_improvement:>8.1f}%")

    print("\n" + "=" * 80)
    print("                         END OF REPORT")
    print("=" * 80)


def plot_performance_charts(passive_results: BacktestResults, spy_results: BacktestResults):
    """
    Create performance visualization charts

    Args:
        passive_results: Results from passive portfolio strategy
        spy_results: Results from SPY benchmark
    """
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Equity Curves
    ax = axes[0, 0]
    passive_curve_normalized = passive_results.equity_curve / passive_results.equity_curve.iloc[0] * 100
    spy_curve_normalized = spy_results.equity_curve / spy_results.equity_curve.iloc[0] * 100

    ax.plot(passive_curve_normalized.index, passive_curve_normalized.values,
            label='Passive Portfolio', linewidth=2, color='blue')
    ax.plot(spy_curve_normalized.index, spy_curve_normalized.values,
            label='SPY Benchmark', linewidth=2, color='gray', alpha=0.7)
    ax.set_title('Portfolio Growth (Base = 100)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (Normalized)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Drawdown Chart
    ax = axes[0, 1]
    passive_cum = (1 + passive_results.daily_returns).cumprod()
    passive_running_max = passive_cum.expanding().max()
    passive_dd = (passive_cum - passive_running_max) / passive_running_max * 100

    spy_cum = (1 + spy_results.daily_returns).cumprod()
    spy_running_max = spy_cum.expanding().max()
    spy_dd = (spy_cum - spy_running_max) / spy_running_max * 100

    ax.fill_between(passive_dd.index, passive_dd.values, 0,
                    alpha=0.3, color='red', label='Passive Portfolio')
    ax.fill_between(spy_dd.index, spy_dd.values, 0,
                    alpha=0.3, color='gray', label='SPY Benchmark')
    ax.set_title('Drawdown Analysis', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Rolling Sharpe Ratio (252-day)
    ax = axes[1, 0]
    window = 252
    passive_rolling_sharpe = (passive_results.daily_returns.rolling(window).mean() * 252) / \
                             (passive_results.daily_returns.rolling(window).std() * np.sqrt(252))
    spy_rolling_sharpe = (spy_results.daily_returns.rolling(window).mean() * 252) / \
                        (spy_results.daily_returns.rolling(window).std() * np.sqrt(252))

    ax.plot(passive_rolling_sharpe.index, passive_rolling_sharpe.values,
            label='Passive Portfolio', linewidth=2, color='blue')
    ax.plot(spy_rolling_sharpe.index, spy_rolling_sharpe.values,
            label='SPY Benchmark', linewidth=2, color='gray', alpha=0.7)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Target (1.0)')
    ax.set_title('Rolling Sharpe Ratio (1-Year)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Monthly Returns Heatmap
    ax = axes[1, 1]
    monthly_returns = passive_results.daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_pivot = pd.pivot_table(
        pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        }),
        values='Return', index='Month', columns='Year'
    )

    sns.heatmap(monthly_pivot, annot=True, fmt='.1f', cmap='RdYlGn',
               center=0, ax=ax, cbar_kws={'label': 'Return (%)'})
    ax.set_title('Monthly Returns Heatmap (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Month')

    plt.suptitle('Passive Portfolio Strategy Performance Analysis (2020-2024)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/waiyang/Desktop/repo/dreamers-v2/src/umtrading/backtest_results.png', dpi=100, bbox_inches='tight')
    plt.show()

    print("\n📊 Performance charts saved to: backtest_results.png")


def main():
    """Main function to run comprehensive backtest"""

    # Configuration
    START_DATE = "2020-01-01"
    END_DATE = "2024-12-31"
    INITIAL_CAPITAL = 100000.0
    COMMISSION = 1.0  # $1 per trade
    SLIPPAGE = 0.001  # 0.1% slippage

    print("\n" + "=" * 80)
    print("         PASSIVE PORTFOLIO STRATEGY - COMPREHENSIVE BACKTEST")
    print("=" * 80)
    print(f"\n⚙️  CONFIGURATION:")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Commission: ${COMMISSION} per trade")
    print(f"  Slippage: {SLIPPAGE * 100}%")
    print(f"  Rebalancing: Quarterly with 2% drift threshold")

    # Initialize strategy
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ']
    print(f"\n📈 PORTFOLIO COMPOSITION:")
    print(f"  Stocks: {', '.join(symbols)}")
    print(f"  Allocation: Equal weight (10% each)")

    # Create passive portfolio strategy
    passive_strategy = PassivePortfolio(
        symbols=symbols,
        rebalance_frequency='Q',
        drift_threshold=0.02,
        initial_capital=INITIAL_CAPITAL
    )

    # Initialize backtesting engine
    backtest_engine = SimpleBacktest(
        start_date=START_DATE,
        end_date=END_DATE,
        initial_capital=INITIAL_CAPITAL,
        commission=COMMISSION,
        slippage=SLIPPAGE,
        risk_free_rate=0.02
    )

    # Fetch data
    print("\n📥 FETCHING MARKET DATA...")
    success = backtest_engine.fetch_data(symbols, benchmark='SPY')
    if not success:
        print("Error: Failed to fetch data")
        return

    # Run passive portfolio backtest
    print("\n🔄 RUNNING PASSIVE PORTFOLIO BACKTEST...")
    passive_results = backtest_engine.run_backtest(passive_strategy)

    # Print passive portfolio results
    backtest_engine.print_results(passive_results, "Passive Portfolio (Equal Weight)")

    # Run SPY benchmark backtest
    print("\n🔄 RUNNING SPY BENCHMARK BACKTEST...")
    spy_results = run_spy_benchmark_backtest(START_DATE, END_DATE, INITIAL_CAPITAL)

    # Print SPY results
    if spy_results:
        backtest_engine.print_results(spy_results, "SPY Buy & Hold Benchmark")

        # Create comprehensive performance report
        create_performance_report(passive_results, spy_results)

        # Generate performance charts
        try:
            plot_performance_charts(passive_results, spy_results)
        except Exception as e:
            print(f"\nWarning: Could not generate charts: {e}")

    # Save results to CSV
    try:
        # Save equity curves
        results_df = pd.DataFrame({
            'Date': passive_results.equity_curve.index,
            'Passive_Portfolio': passive_results.equity_curve.values,
            'SPY_Benchmark': spy_results.equity_curve.values if spy_results else None,
            'Passive_Daily_Return': passive_results.daily_returns.values,
            'SPY_Daily_Return': spy_results.daily_returns.values if spy_results else None
        })
        results_df.to_csv('/Users/waiyang/Desktop/repo/dreamers-v2/src/umtrading/backtest_results.csv', index=False)
        print("\n📁 Results saved to: backtest_results.csv")
    except Exception as e:
        print(f"\nWarning: Could not save results to CSV: {e}")

    print("\n✅ BACKTEST COMPLETE!")


if __name__ == "__main__":
    main()