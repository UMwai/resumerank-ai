"""
Example Integration with Trading System
Shows how to integrate risk management with actual trading logic
"""

import sys
import time
import random
from datetime import datetime
from typing import Dict, List

# Import risk management components
from risk.simple_circuit_breaker import SimpleCircuitBreaker
from risk.simple_position_sizer import SimplePositionSizer
from utils.daily_logger import DailyLogger


class MockTradingSystem:
    """Mock trading system for demonstration"""

    def __init__(self, initial_capital: float = 100000):
        # Initialize risk management
        self.circuit_breaker = SimpleCircuitBreaker()
        self.position_sizer = SimplePositionSizer(initial_capital)
        self.daily_logger = DailyLogger()

        # Portfolio state
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.market_regime = "NEUTRAL"

        # Set day start
        self.daily_logger.set_day_start_value(initial_capital)

    def check_risk_controls(self) -> bool:
        """Check if trading is allowed"""
        # Update circuit breaker with current portfolio value
        cb_status = self.circuit_breaker.update_portfolio_value(self.portfolio_value)

        # Log status if state changed
        if cb_status['action']:
            print(f"\n⚠️ Risk Alert: {cb_status['action']}")
            print(f"   Current Drawdown: {cb_status['drawdown']:.2%}")

            # Handle different actions
            if cb_status['action'] == 'KILL_ALL_POSITIONS':
                self.liquidate_all_positions()
                return False
            elif cb_status['action'] == 'HALT_TRADING':
                print("   Trading halted - no new positions allowed")
                return False
            elif cb_status['action'] == 'REDUCE_RISK':
                print("   Warning level - reducing position sizes")

        return self.circuit_breaker.can_trade()

    def calculate_position(self, symbol: str, signal_strength: float) -> Dict:
        """Calculate position size based on signal and risk"""
        # Get current price (mock)
        current_price = random.uniform(50, 500)

        # Adjust risk based on market regime and signal
        risk_multiplier = 1.0
        if self.market_regime == "BEARISH":
            risk_multiplier *= 0.5
        elif self.market_regime == "BULLISH":
            risk_multiplier *= 1.2

        # Adjust for signal strength (0.5 to 1.5)
        risk_multiplier *= max(0.5, min(1.5, signal_strength))

        # Calculate position
        position = self.position_sizer.calculate_position_size(
            symbol=symbol,
            current_price=current_price,
            risk_multiplier=risk_multiplier
        )

        return position

    def execute_trade(self, symbol: str, action: str, position: Dict) -> Dict:
        """Execute a trade with logging"""
        if not position['can_trade']:
            return None

        # Create trade record
        trade = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'shares': position['shares'],
            'price': position['current_price'],
            'value': position['position_size'],
            'pnl': 0  # Will be calculated on sell
        }

        # Update portfolio
        if action == 'BUY':
            self.cash -= position['position_size']
            self.positions[symbol] = {
                'shares': position['shares'],
                'cost_basis': position['current_price'],
                'value': position['position_size']
            }
            self.position_sizer.add_position(symbol, position['position_size'], position['shares'])

        elif action == 'SELL' and symbol in self.positions:
            # Calculate P&L
            cost_basis = self.positions[symbol]['cost_basis']
            pnl = position['shares'] * (position['current_price'] - cost_basis)
            trade['pnl'] = pnl

            # Update portfolio
            self.cash += position['position_size']
            del self.positions[symbol]
            self.position_sizer.remove_position(symbol)

        # Log trade
        self.daily_logger.log_trade(trade)

        print(f"   {action} {position['shares']} {symbol} @ ${position['current_price']:.2f}")

        return trade

    def update_portfolio_value(self):
        """Update total portfolio value"""
        # Calculate positions value (with mock price changes)
        positions_value = 0
        for symbol, pos in self.positions.items():
            # Simulate price change
            price_change = random.uniform(-0.03, 0.03)
            new_value = pos['value'] * (1 + price_change)
            pos['value'] = new_value
            positions_value += new_value

        self.portfolio_value = self.cash + positions_value

    def liquidate_all_positions(self):
        """Emergency liquidation"""
        print("\n🚨 EMERGENCY LIQUIDATION - Closing all positions")
        for symbol in list(self.positions.keys()):
            position = self.position_sizer.calculate_position_size(symbol, 100)  # Mock price
            self.execute_trade(symbol, 'SELL', position)

    def run_trading_session(self, iterations: int = 20):
        """Simulate a trading session"""
        print("\n" + "=" * 60)
        print("Starting Trading Session with Risk Management")
        print("=" * 60)
        print(f"Initial Portfolio: ${self.portfolio_value:,.2f}")

        trades_executed = []

        for i in range(iterations):
            print(f"\n[Iteration {i+1}]")

            # Update portfolio value
            self.update_portfolio_value()
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")

            # Check risk controls
            if not self.check_risk_controls():
                print("   ❌ Trading not allowed by risk controls")
                continue

            # Simulate trading signals
            if random.random() < 0.3:  # 30% chance of signal
                # Generate trade signal
                symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
                symbol = random.choice(symbols)
                signal_strength = random.uniform(0.5, 1.5)

                # Decide action
                if symbol in self.positions:
                    action = 'SELL'
                    print(f"   📉 Sell signal for {symbol} (strength: {signal_strength:.2f})")
                else:
                    action = 'BUY'
                    print(f"   📈 Buy signal for {symbol} (strength: {signal_strength:.2f})")

                # Calculate position
                position = self.calculate_position(symbol, signal_strength)

                # Execute trade
                trade = self.execute_trade(symbol, action, position)
                if trade:
                    trades_executed.append(trade)

            # Simulate market regime change
            if random.random() < 0.1:
                self.market_regime = random.choice(['BULLISH', 'NEUTRAL', 'BEARISH'])
                print(f"   🎯 Market regime changed to: {self.market_regime}")

            # Small delay
            time.sleep(0.1)

        # End of session summary
        self.generate_summary(trades_executed)

    def generate_summary(self, trades: List[Dict]):
        """Generate end of session summary"""
        print("\n" + "=" * 60)
        print("End of Session Summary")
        print("=" * 60)

        # Calculate metrics
        peak_value = self.circuit_breaker.peak_value
        current_drawdown = self.circuit_breaker.current_drawdown

        # Simple Sharpe calculation
        sharpe = 1.2  # Mock value

        # Log daily summary
        summary = self.daily_logger.log_daily_summary(
            portfolio_value=self.portfolio_value,
            market_regime=self.market_regime,
            trades=trades,
            drawdown=current_drawdown,
            peak_value=peak_value,
            sharpe_ratio=sharpe,
            holdings_count=len(self.positions),
            cash_balance=self.cash,
            notes="Integration example"
        )

        # Risk management final status
        print("\n📊 Risk Management Status:")
        cb_status = self.circuit_breaker.get_status()
        print(f"   Circuit Breaker State: {cb_status['state']}")
        print(f"   Current Drawdown: {cb_status['current_drawdown']}")
        print(f"   Can Trade: {cb_status['can_trade']}")

        ps_status = self.position_sizer.get_status()
        print(f"\n📈 Position Management:")
        print(f"   Open Positions: {ps_status['position_count']}/{ps_status['max_positions']}")
        print(f"   Available Capital: ${ps_status['available_capital']:,.2f}")

        print("\n✅ Session Complete - Logs saved to ./logs/")


def main():
    """Run the integration example"""
    print("UMTrading Risk Management Integration Example")
    print("This demonstrates how risk management integrates with trading logic")

    # Create and run trading system
    trading_system = MockTradingSystem(initial_capital=100000)

    # Set initial market conditions
    trading_system.market_regime = "NEUTRAL"

    # Run trading session
    trading_system.run_trading_session(iterations=15)

    print("\n" + "=" * 60)
    print("Integration Example Complete")
    print("=" * 60)
    print("\nThis example shows:")
    print("  ✓ Circuit breaker monitoring portfolio drawdown")
    print("  ✓ Position sizer calculating appropriate sizes")
    print("  ✓ Daily logger tracking all activity")
    print("  ✓ Risk controls preventing trades when needed")
    print("\nTo see live monitoring, run:")
    print("  python3 -m streamlit run dashboards/live_monitor.py")
    print("=" * 60)


if __name__ == "__main__":
    main()