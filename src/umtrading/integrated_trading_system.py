#!/usr/bin/env python3
"""
Integrated Trading System - Unified Orchestration
Combines: Data Pipeline + Regime Detection + Strategy + Risk Management

Usage:
    python integrated_trading_system.py --mode backtest
    python integrated_trading_system.py --mode paper
    python integrated_trading_system.py --mode live --capital 10000
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path

# Import all 3 tracks
from data.simple_data_collector import SimpleDataCollector
from regime.simple_detector import SimpleRegimeDetector
from strategies.passive_portfolio import PassivePortfolio
from backtesting.simple_backtest import SimpleBacktester
from risk.simple_circuit_breaker import SimpleCircuitBreaker
from risk.simple_position_sizer import SimplePositionSizer
from utils.daily_logger import DailyLogger


class IntegratedTradingSystem:
    """
    Unified trading system combining all 3 tracks:
    - Track 1: Data + Regime Detection
    - Track 2: Portfolio Strategy + Backtesting
    - Track 3: Risk Management + Monitoring
    """

    def __init__(self, capital=10000, mode="paper"):
        """
        Initialize trading system

        Args:
            capital: Starting capital in USD
            mode: "backtest" | "paper" | "live"
        """
        self.capital = capital
        self.mode = mode
        self.initial_capital = capital

        # Track 1: Data + Regime
        self.data_collector = SimpleDataCollector()
        self.regime_detector = SimpleRegimeDetector()

        # Track 2: Strategy
        self.portfolio = PassivePortfolio(capital)

        # Track 3: Risk Management
        self.circuit_breaker = SimpleCircuitBreaker(capital)
        self.position_sizer = SimplePositionSizer(max_positions=10, max_pct_per_position=0.02)
        self.logger = DailyLogger()

        # State
        self.current_regime = "UNKNOWN"
        self.positions = {}
        self.cash = capital
        self.peak_value = capital
        self.portfolio_history = []

    def run_backtest(self, start_date="2020-01-01", end_date="2024-11-22"):
        """
        Run historical backtest to validate strategy

        Returns:
            dict: Performance metrics
        """
        print(f"\n{'='*70}")
        print(f"  RUNNING BACKTEST: {start_date} to {end_date}")
        print(f"  Starting Capital: ${self.capital:,.0f}")
        print(f"{'='*70}\n")

        # Use existing backtesting engine from Track 2
        backtester = SimpleBacktester()
        results = backtester.run_backtest(
            strategy=self.portfolio,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.capital
        )

        print(f"\n{'='*70}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*70}")
        print(f"  Total Return:     {results['total_return']:>10.2%}")
        print(f"  Annualized Return:{results['annualized_return']:>10.2%}")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:>10.2f}")
        print(f"  Max Drawdown:     {results['max_drawdown']:>10.2%}")
        print(f"  Alpha vs SPY:     {results['alpha_vs_spy']:>10.2%}")
        print(f"{'='*70}\n")

        # Validate against targets
        self._validate_backtest_results(results)

        return results

    def _validate_backtest_results(self, results):
        """Check if backtest meets minimum criteria"""
        print("\n📊 VALIDATION CHECKS:")

        checks = [
            ("Sharpe Ratio > 1.0", results['sharpe_ratio'] > 1.0),
            ("Max Drawdown < 40%", results['max_drawdown'] > -0.40),
            ("Alpha vs SPY > 0%", results['alpha_vs_spy'] > 0),
            ("Total Return > 100%", results['total_return'] > 1.0)
        ]

        all_passed = True
        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: {check_name}")
            if not passed:
                all_passed = False

        if all_passed:
            print("\n🎉 ALL VALIDATION CHECKS PASSED - Ready for paper trading!\n")
        else:
            print("\n⚠️  SOME CHECKS FAILED - Review strategy before deployment\n")

        return all_passed

    def run_paper_trading(self, duration_days=30):
        """
        Run paper trading for specified duration

        Args:
            duration_days: Number of days to paper trade
        """
        print(f"\n{'='*70}")
        print(f"  PAPER TRADING MODE - {duration_days} Days")
        print(f"  Starting Capital: ${self.capital:,.0f}")
        print(f"{'='*70}\n")

        start_date = datetime.now() - timedelta(days=duration_days)

        # Download data for paper trading period
        spy_data, vix_data = self.data_collector.fetch_historical_data(
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d')
        )

        # Simulate each trading day
        for date in pd.date_range(start_date, datetime.now(), freq='B'):  # Business days
            print(f"\n📅 {date.strftime('%Y-%m-%d')}")

            # 1. Detect market regime
            regime = self.regime_detector.detect_regime(
                spy_data.loc[:date],
                vix_data.loc[:date]
            )
            print(f"   Market Regime: {regime}")

            # 2. Check circuit breakers
            portfolio_value = self._calculate_portfolio_value(date)
            breaker_status = self.circuit_breaker.check_drawdown(portfolio_value)

            if breaker_status in ["HALT", "KILL"]:
                print(f"   🚨 {breaker_status}: Trading halted due to drawdown")
                continue

            # 3. Check if rebalancing needed
            current_prices = self._get_prices_on_date(date)
            if self.portfolio.should_rebalance(self.positions, current_prices):
                print(f"   🔄 Rebalancing portfolio...")
                trades = self.portfolio.get_rebalance_trades(
                    current_positions=self.positions,
                    current_prices=current_prices,
                    available_capital=portfolio_value
                )

                # Execute trades (paper mode)
                for trade in trades:
                    print(f"      {trade['action']} {trade['shares']} {trade['symbol']} @ ${trade['price']:.2f}")

                # Update positions
                self._execute_trades(trades, current_prices)

            # 4. Log daily summary
            self.logger.log_daily_summary(
                date=date,
                portfolio_value=portfolio_value,
                regime=regime,
                positions=self.positions,
                cash=self.cash
            )

            # Update history
            self.portfolio_history.append({
                'date': date,
                'value': portfolio_value,
                'regime': regime
            })

        # Final summary
        self._print_paper_trading_summary()

    def run_live_trading(self):
        """
        Run live trading with real broker API

        IMPORTANT: This connects to real broker and executes real trades!
        """
        print(f"\n{'='*70}")
        print(f"  ⚠️  LIVE TRADING MODE - REAL MONEY")
        print(f"  Starting Capital: ${self.capital:,.0f}")
        print(f"{'='*70}\n")

        print("⚠️  Live trading requires broker API setup")
        print("Next steps:")
        print("1. Set up Interactive Brokers or Alpaca account")
        print("2. Configure API keys in .env file")
        print("3. Test with paper trading account first")
        print("4. Enable live trading mode")
        print("\nFor now, run in paper trading mode to validate.\n")

    def run_realtime_monitor(self):
        """
        Launch real-time monitoring dashboard
        """
        print("\n🚀 Launching real-time monitoring dashboard...")
        print("   Dashboard will open in your browser at http://localhost:8501")
        print("   Press Ctrl+C to stop\n")

        import subprocess
        subprocess.run([
            "streamlit", "run",
            "dashboards/live_monitor.py",
            "--server.port", "8501"
        ])

    def _calculate_portfolio_value(self, date):
        """Calculate current portfolio value"""
        holdings_value = 0
        for symbol, shares in self.positions.items():
            # Get price on date
            price = self._get_price(symbol, date)
            holdings_value += shares * price

        return self.cash + holdings_value

    def _get_prices_on_date(self, date):
        """Get prices for all holdings on specific date"""
        prices = {}
        for symbol in self.portfolio.HOLDINGS:
            prices[symbol] = self._get_price(symbol, date)
        return prices

    def _get_price(self, symbol, date):
        """Get closing price for symbol on date"""
        # Simplified - in production would fetch from data store
        import yfinance as yf
        data = yf.download(symbol, start=date, end=date + timedelta(days=1), progress=False)
        if len(data) > 0:
            return data['Close'].iloc[0]
        return None

    def _execute_trades(self, trades, current_prices):
        """Execute trades and update positions"""
        for trade in trades:
            symbol = trade['symbol']
            shares = trade['shares']
            action = trade['action']
            price = current_prices[symbol]

            if action == 'BUY':
                cost = shares * price * 1.001 + 1  # Slippage + commission
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares

            elif action == 'SELL':
                proceeds = shares * price * 0.999 - 1  # Slippage - commission
                self.cash += proceeds
                self.positions[symbol] = self.positions.get(symbol, 0) - shares

                if self.positions[symbol] <= 0:
                    del self.positions[symbol]

    def _print_paper_trading_summary(self):
        """Print summary of paper trading results"""
        if not self.portfolio_history:
            return

        history_df = pd.DataFrame(self.portfolio_history)

        final_value = history_df['value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Calculate Sharpe
        returns = history_df['value'].pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # Calculate max drawdown
        peak = history_df['value'].expanding().max()
        drawdown = (history_df['value'] - peak) / peak
        max_drawdown = drawdown.min()

        print(f"\n{'='*70}")
        print(f"  PAPER TRADING SUMMARY")
        print(f"{'='*70}")
        print(f"  Duration:         {len(history_df)} days")
        print(f"  Final Value:      ${final_value:,.0f}")
        print(f"  Total Return:     {total_return:>10.2%}")
        print(f"  Sharpe Ratio:     {sharpe:>10.2f}")
        print(f"  Max Drawdown:     {max_drawdown:>10.2%}")
        print(f"{'='*70}\n")

        # Validation
        if sharpe > 0.8 and max_drawdown > -0.15:
            print("✅ PAPER TRADING SUCCESSFUL - Ready for live trading!")
        else:
            print("⚠️  Review results before going live")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Integrated Trading System")
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live', 'monitor'],
                       default='backtest', help='Trading mode')
    parser.add_argument('--capital', type=int, default=10000,
                       help='Starting capital in USD')
    parser.add_argument('--days', type=int, default=30,
                       help='Paper trading duration in days')

    args = parser.parse_args()

    # Initialize system
    system = IntegratedTradingSystem(capital=args.capital, mode=args.mode)

    # Run selected mode
    if args.mode == 'backtest':
        results = system.run_backtest()

    elif args.mode == 'paper':
        system.run_paper_trading(duration_days=args.days)

    elif args.mode == 'live':
        confirm = input("⚠️  WARNING: This will trade REAL MONEY. Type 'YES' to confirm: ")
        if confirm == 'YES':
            system.run_live_trading()
        else:
            print("Live trading cancelled.")

    elif args.mode == 'monitor':
        system.run_realtime_monitor()


if __name__ == '__main__':
    main()
