"""
Demo Script for Risk Management & Monitoring
Shows integration of all Track 3 components
"""

import sys
import time
import random
import numpy as np
from datetime import datetime, timedelta

# Import our risk management modules
from risk.simple_circuit_breaker import SimpleCircuitBreaker
from risk.simple_position_sizer import SimplePositionSizer
from utils.daily_logger import DailyLogger


def simulate_trading_day():
    """
    Simulate a trading day with risk management
    """
    print("\n" + "=" * 60)
    print("UMTrading Risk Management Demo")
    print("=" * 60)

    # Initialize components
    portfolio_value = 100000
    circuit_breaker = SimpleCircuitBreaker()
    position_sizer = SimplePositionSizer(portfolio_value)
    daily_logger = DailyLogger(log_dir="./logs")

    # Set day start
    daily_logger.set_day_start_value(portfolio_value)

    print(f"\nStarting Portfolio Value: ${portfolio_value:,.2f}")
    print("-" * 60)

    # Simulate market conditions
    market_regimes = ["BULLISH", "NEUTRAL", "BEARISH"]
    current_regime = random.choice(market_regimes)
    print(f"Market Regime: {current_regime}\n")

    # Simulate portfolio value changes throughout the day
    portfolio_values = []
    trades = []

    # Generate realistic intraday portfolio movements
    hours = 7  # Trading hours
    intervals = hours * 4  # 15-minute intervals

    print("Simulating trading day...")
    print("-" * 60)

    for i in range(intervals):
        # Simulate portfolio value change
        if current_regime == "BULLISH":
            change = np.random.normal(0.001, 0.005)
        elif current_regime == "BEARISH":
            change = np.random.normal(-0.002, 0.008)
        else:
            change = np.random.normal(0, 0.004)

        portfolio_value *= (1 + change)
        portfolio_values.append(portfolio_value)

        # Update circuit breaker
        cb_status = circuit_breaker.update_portfolio_value(portfolio_value)

        # Check for circuit breaker triggers
        if cb_status['action']:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
                  f"Circuit Breaker: {cb_status['action']}")
            print(f"Portfolio: ${portfolio_value:,.2f} | "
                  f"Drawdown: {cb_status['drawdown']:.2%}")

        # Simulate occasional trades
        if random.random() < 0.1 and circuit_breaker.can_trade():
            # Pick a random stock
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
            symbol = random.choice(symbols)
            price = random.uniform(100, 500)

            # Calculate position size
            position = position_sizer.calculate_position_size(
                symbol=symbol,
                current_price=price,
                risk_multiplier=1.0 if current_regime != "BEARISH" else 0.5
            )

            if position['can_trade']:
                # Execute trade
                action = random.choice(['BUY', 'SELL'])
                trade = {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'action': action,
                    'shares': position['shares'],
                    'price': price,
                    'pnl': random.uniform(-200, 400) if action == 'SELL' else 0
                }
                trades.append(trade)
                daily_logger.log_trade(trade)

                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"{action} {position['shares']} {symbol} @ ${price:.2f}")

                if action == 'BUY':
                    position_sizer.add_position(symbol, position['position_size'], position['shares'])
                else:
                    position_sizer.remove_position(symbol)

        # Small delay for simulation
        time.sleep(0.01)

    print("\n" + "=" * 60)
    print("End of Day Summary")
    print("=" * 60)

    # Calculate final metrics
    final_value = portfolio_values[-1]
    daily_pnl = final_value - 100000
    peak_value = max(portfolio_values)
    current_drawdown = (peak_value - final_value) / peak_value if peak_value > 0 else 0

    # Calculate simple Sharpe ratio
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-6)

    # Log daily summary
    summary = daily_logger.log_daily_summary(
        portfolio_value=final_value,
        market_regime=current_regime,
        trades=trades,
        drawdown=current_drawdown,
        peak_value=peak_value,
        sharpe_ratio=sharpe,
        holdings_count=position_sizer.position_count,
        cash_balance=position_sizer.get_available_capital(),
        notes="Demo simulation"
    )

    # Print final status
    print("\n" + "=" * 60)
    print("Risk Management Status")
    print("=" * 60)

    # Circuit breaker status
    cb_final = circuit_breaker.get_status()
    print(f"\nCircuit Breaker:")
    print(f"  State: {cb_final['state']}")
    print(f"  Can Trade: {cb_final['can_trade']}")
    print(f"  Current Drawdown: {cb_final['current_drawdown']}")

    # Position sizing status
    ps_final = position_sizer.get_status()
    print(f"\nPosition Sizer:")
    print(f"  Positions: {ps_final['position_count']}/{ps_final['max_positions']}")
    print(f"  Total Allocated: ${ps_final['total_allocated']:,.2f}")
    print(f"  Available Capital: ${ps_final['available_capital']:,.2f}")

    print("\n" + "=" * 60)
    print(f"Log files saved to: {daily_logger.log_dir}/")
    print("=" * 60)

    return summary


def run_dashboard_instructions():
    """Print instructions for running the dashboard"""
    print("\n" + "=" * 60)
    print("To Launch the Live Monitoring Dashboard:")
    print("=" * 60)
    print("\n1. Install requirements:")
    print("   pip install -r requirements.txt")
    print("\n2. Run the dashboard:")
    print("   streamlit run dashboards/live_monitor.py")
    print("\n3. Open your browser to: http://localhost:8501")
    print("\nThe dashboard will show:")
    print("  - Real-time portfolio value and P&L")
    print("  - Current market regime")
    print("  - Drawdown monitoring with circuit breaker status")
    print("  - Rolling Sharpe ratio")
    print("  - Holdings table with P&L")
    print("  - Equity curve vs SPY benchmark")
    print("  - Recent trades log")
    print("=" * 60)


if __name__ == "__main__":
    # Run the simulation
    simulate_trading_day()

    # Show dashboard instructions
    run_dashboard_instructions()