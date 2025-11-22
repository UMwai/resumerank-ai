"""
Passive Portfolio Strategy Module
Implements equal-weight portfolio with quarterly rebalancing for top S&P 100 stocks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class PassivePortfolio:
    """
    Passive equal-weight portfolio strategy for top 10 S&P 100 stocks
    """

    def __init__(self,
                 symbols: Optional[List[str]] = None,
                 target_weights: Optional[Dict[str, float]] = None,
                 rebalance_frequency: str = 'Q',  # Q=Quarterly, M=Monthly, Y=Yearly
                 drift_threshold: float = 0.02,  # 2% drift threshold
                 initial_capital: float = 100000.0):
        """
        Initialize passive portfolio strategy

        Args:
            symbols: List of stock symbols (default: top 10 S&P 100)
            target_weights: Target weight for each symbol (default: equal weight)
            rebalance_frequency: Rebalancing frequency ('Q', 'M', 'Y')
            drift_threshold: Threshold for drift-based rebalancing (0.02 = 2%)
            initial_capital: Initial capital for portfolio
        """
        # Default top 10 S&P 100 stocks
        self.symbols = symbols or [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ'
        ]

        # Equal weight allocation by default
        if target_weights is None:
            equal_weight = 1.0 / len(self.symbols)
            self.target_weights = {symbol: equal_weight for symbol in self.symbols}
        else:
            self.target_weights = target_weights

        self.rebalance_frequency = rebalance_frequency
        self.drift_threshold = drift_threshold
        self.initial_capital = initial_capital

        # Track portfolio state
        self.current_positions = {symbol: 0 for symbol in self.symbols}
        self.current_weights = {symbol: 0.0 for symbol in self.symbols}
        self.cash = initial_capital
        self.portfolio_value = initial_capital

        # Track rebalancing dates
        self.last_rebalance_date = None
        self.rebalance_dates = []

    def should_rebalance(self, date: pd.Timestamp, prices: Dict[str, float]) -> bool:
        """
        Determine if portfolio should be rebalanced

        Args:
            date: Current date
            prices: Current prices for all symbols

        Returns:
            True if rebalancing is needed, False otherwise
        """
        # First trade - always rebalance
        if self.last_rebalance_date is None:
            return True

        # Check periodic rebalancing
        if self.rebalance_frequency == 'Q':
            # Quarterly rebalancing - check if new quarter
            if date.quarter != self.last_rebalance_date.quarter or \
               date.year != self.last_rebalance_date.year:
                return True
        elif self.rebalance_frequency == 'M':
            # Monthly rebalancing
            if date.month != self.last_rebalance_date.month or \
               date.year != self.last_rebalance_date.year:
                return True
        elif self.rebalance_frequency == 'Y':
            # Yearly rebalancing
            if date.year != self.last_rebalance_date.year:
                return True

        # Check drift-based rebalancing
        if self._calculate_max_drift(prices) > self.drift_threshold:
            return True

        return False

    def _calculate_max_drift(self, prices: Dict[str, float]) -> float:
        """
        Calculate maximum drift from target weights

        Args:
            prices: Current prices for all symbols

        Returns:
            Maximum absolute drift from target weights
        """
        # Calculate current portfolio value
        portfolio_value = self.cash
        for symbol in self.symbols:
            if symbol in prices:
                portfolio_value += self.current_positions[symbol] * prices[symbol]

        if portfolio_value <= 0:
            return 0.0

        # Calculate current weights and drift
        max_drift = 0.0
        for symbol in self.symbols:
            if symbol in prices:
                current_value = self.current_positions[symbol] * prices[symbol]
                current_weight = current_value / portfolio_value
                target_weight = self.target_weights[symbol]
                drift = abs(current_weight - target_weight)
                max_drift = max(max_drift, drift)

        return max_drift

    def calculate_rebalance_trades(self,
                                  date: pd.Timestamp,
                                  prices: Dict[str, float],
                                  commission: float = 1.0,
                                  slippage: float = 0.001) -> Dict[str, int]:
        """
        Calculate trades needed to rebalance portfolio

        Args:
            date: Current date
            prices: Current prices for all symbols
            commission: Commission per trade in dollars
            slippage: Slippage as percentage (0.001 = 0.1%)

        Returns:
            Dictionary of trades (positive = buy, negative = sell)
        """
        trades = {}

        # Calculate current portfolio value
        portfolio_value = self.cash
        for symbol in self.symbols:
            if symbol in prices:
                portfolio_value += self.current_positions[symbol] * prices[symbol]

        self.portfolio_value = portfolio_value

        # Account for transaction costs
        # Reserve cash for commissions (estimate max trades)
        reserved_cash = commission * len(self.symbols) * 2
        investable_value = portfolio_value - reserved_cash

        if investable_value <= 0:
            return {}

        # Calculate target positions
        target_positions = {}
        for symbol in self.symbols:
            if symbol in prices:
                target_value = investable_value * self.target_weights[symbol]
                # Account for slippage in execution price
                execution_price = prices[symbol] * (1 + slippage)
                target_positions[symbol] = int(target_value / execution_price)
            else:
                target_positions[symbol] = 0

        # Calculate trades needed
        for symbol in self.symbols:
            current_pos = self.current_positions[symbol]
            target_pos = target_positions.get(symbol, 0)
            trade_size = target_pos - current_pos

            if trade_size != 0:
                trades[symbol] = trade_size

        return trades

    def execute_trades(self,
                      trades: Dict[str, int],
                      prices: Dict[str, float],
                      date: pd.Timestamp,
                      commission: float = 1.0,
                      slippage: float = 0.001) -> Dict:
        """
        Execute trades and update portfolio state

        Args:
            trades: Dictionary of trades to execute
            prices: Current prices for execution
            date: Execution date
            commission: Commission per trade in dollars
            slippage: Slippage as percentage

        Returns:
            Execution report with details
        """
        execution_report = {
            'date': date,
            'trades': {},
            'costs': {
                'commission': 0.0,
                'slippage': 0.0,
                'total': 0.0
            },
            'cash_before': self.cash,
            'cash_after': 0.0,
            'portfolio_value_before': self.portfolio_value,
            'portfolio_value_after': 0.0
        }

        total_commission = 0.0
        total_slippage = 0.0

        # Execute trades
        for symbol, trade_size in trades.items():
            if trade_size == 0:
                continue

            if symbol not in prices:
                continue

            # Calculate execution price with slippage
            base_price = prices[symbol]
            if trade_size > 0:  # Buy - pay more
                execution_price = base_price * (1 + slippage)
            else:  # Sell - receive less
                execution_price = base_price * (1 - slippage)

            # Calculate trade value and costs
            trade_value = abs(trade_size) * execution_price
            slippage_cost = abs(trade_size) * base_price * slippage

            # Check if we have enough cash for buys
            if trade_size > 0:
                required_cash = trade_value + commission
                if required_cash > self.cash:
                    # Reduce trade size to what we can afford
                    affordable_shares = int((self.cash - commission) / execution_price)
                    if affordable_shares <= 0:
                        continue
                    trade_size = affordable_shares
                    trade_value = trade_size * execution_price
                    slippage_cost = trade_size * base_price * slippage

            # Execute trade
            if trade_size > 0:  # Buy
                self.cash -= (trade_value + commission)
                self.current_positions[symbol] += trade_size
            else:  # Sell
                self.cash += (abs(trade_value) - commission)
                self.current_positions[symbol] += trade_size  # trade_size is negative

            # Track costs
            total_commission += commission
            total_slippage += slippage_cost

            # Record trade
            execution_report['trades'][symbol] = {
                'size': trade_size,
                'price': base_price,
                'execution_price': execution_price,
                'value': trade_value if trade_size > 0 else -trade_value,
                'commission': commission,
                'slippage': slippage_cost
            }

        # Update portfolio state
        self.last_rebalance_date = date
        self.rebalance_dates.append(date)

        # Calculate new portfolio value and weights
        new_portfolio_value = self.cash
        for symbol in self.symbols:
            if symbol in prices:
                position_value = self.current_positions[symbol] * prices[symbol]
                new_portfolio_value += position_value
                self.current_weights[symbol] = position_value / new_portfolio_value if new_portfolio_value > 0 else 0.0

        self.portfolio_value = new_portfolio_value

        # Update execution report
        execution_report['costs']['commission'] = total_commission
        execution_report['costs']['slippage'] = total_slippage
        execution_report['costs']['total'] = total_commission + total_slippage
        execution_report['cash_after'] = self.cash
        execution_report['portfolio_value_after'] = new_portfolio_value

        return execution_report

    def get_positions(self) -> Dict[str, int]:
        """Get current positions"""
        return self.current_positions.copy()

    def get_weights(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        return self.current_weights.copy()

    def get_portfolio_value(self, prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value

        Args:
            prices: Current prices for all symbols

        Returns:
            Total portfolio value
        """
        total_value = self.cash
        for symbol, position in self.current_positions.items():
            if symbol in prices:
                total_value += position * prices[symbol]
        return total_value

    def reset(self):
        """Reset portfolio to initial state"""
        self.current_positions = {symbol: 0 for symbol in self.symbols}
        self.current_weights = {symbol: 0.0 for symbol in self.symbols}
        self.cash = self.initial_capital
        self.portfolio_value = self.initial_capital
        self.last_rebalance_date = None
        self.rebalance_dates = []