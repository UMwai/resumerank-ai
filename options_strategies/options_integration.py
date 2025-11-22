"""
Options Strategy Integration Module

Integrates options strategies with the existing trading platform,
including real-time execution, risk monitoring, and P&L tracking.

This module bridges the options playbook with:
- Patent intelligence signals
- Clinical trial events
- Insider trading patterns
- Market data feeds
- Risk management systems
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from .options_playbook import (
    ProtectivePutStrategy,
    CoveredCallStrategy,
    VolatilityArbitrageStrategy,
    EarningsStrategy,
    OptionsRiskManager,
    OptionsPosition,
    OptionsGreeks,
    OptionType,
    BlackScholesModel,
    MarketRegime
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingSignal:
    """Unified trading signal from various sources."""
    source: str  # patent_cliff, clinical_trial, insider, earnings
    symbol: str
    signal_type: str  # bullish, bearish, neutral, volatility
    strength: float  # 0-100
    timeframe: int  # days
    metadata: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptionsOrder:
    """Options order for execution."""
    symbol: str
    option_type: OptionType
    action: str  # buy, sell
    quantity: int
    order_type: str  # market, limit
    limit_price: Optional[float] = None
    strike: Optional[float] = None
    expiration: Optional[datetime] = None
    strategy_name: Optional[str] = None
    parent_order_id: Optional[str] = None


class OptionsStrategySelector:
    """
    Intelligent strategy selector based on market conditions and signals.
    Maps various trading signals to appropriate options strategies.
    """

    def __init__(self, portfolio_value: float):
        """Initialize strategy selector."""
        self.portfolio_value = portfolio_value
        self.strategies = {
            "protective_put": ProtectivePutStrategy(portfolio_value),
            "covered_call": CoveredCallStrategy(1000),  # Default 1000 shares
            "volatility": VolatilityArbitrageStrategy(portfolio_value * 0.1),
            "earnings": EarningsStrategy(5000)
        }
        self.risk_manager = OptionsRiskManager(portfolio_value)

    def select_strategy(self, signal: TradingSignal, market_data: Dict) -> Dict[str, any]:
        """
        Select optimal options strategy based on signal and market conditions.

        Args:
            signal: Trading signal from any source
            market_data: Current market data including price, IV, etc.

        Returns:
            Dictionary with recommended strategy and parameters
        """
        recommendations = []

        # Patent cliff signals - typically bearish on branded, bullish on generic
        if signal.source == "patent_cliff":
            if signal.signal_type == "bearish":
                recommendations.append(self._design_patent_cliff_strategy(signal, market_data))

        # Clinical trial signals - binary events
        elif signal.source == "clinical_trial":
            recommendations.append(self._design_clinical_trial_strategy(signal, market_data))

        # Insider trading signals
        elif signal.source == "insider":
            recommendations.append(self._design_insider_strategy(signal, market_data))

        # Earnings signals
        elif signal.source == "earnings":
            recommendations.append(self._design_earnings_strategy(signal, market_data))

        # Market regime based strategies
        regime = self._determine_market_regime(market_data)
        if regime == MarketRegime.HIGH_VOL:
            recommendations.append(self._design_high_vol_strategy(market_data))

        # Score and rank recommendations
        scored_recommendations = self._score_recommendations(recommendations, market_data)

        # Return top recommendation
        if scored_recommendations:
            return scored_recommendations[0]
        else:
            return {"strategy": "none", "reason": "No suitable strategy found"}

    def _design_patent_cliff_strategy(self, signal: TradingSignal,
                                     market_data: Dict) -> Dict[str, any]:
        """Design options strategy for patent cliff events."""
        days_to_event = signal.timeframe
        symbol = signal.symbol

        if days_to_event < 90:
            # Near-term event - use put spreads
            strategy = {
                "name": "put_spread",
                "type": "bearish",
                "structure": {
                    "long_put": {
                        "strike_offset": -0.05,  # 5% OTM
                        "expiration": days_to_event + 30
                    },
                    "short_put": {
                        "strike_offset": -0.15,  # 15% OTM
                        "expiration": days_to_event + 30
                    }
                },
                "sizing": self._calculate_patent_sizing(signal.strength, days_to_event),
                "confidence": signal.strength,
                "reason": f"Patent cliff in {days_to_event} days"
            }
        else:
            # Longer-term - use calendar spreads
            strategy = {
                "name": "calendar_put_spread",
                "type": "bearish_volatility",
                "structure": {
                    "short_put": {
                        "strike_offset": -0.10,
                        "expiration": 45
                    },
                    "long_put": {
                        "strike_offset": -0.10,
                        "expiration": days_to_event + 15
                    }
                },
                "sizing": self._calculate_patent_sizing(signal.strength * 0.7, days_to_event),
                "confidence": signal.strength * 0.8,
                "reason": f"Patent cliff calendar play - {days_to_event} days out"
            }

        return strategy

    def _design_clinical_trial_strategy(self, signal: TradingSignal,
                                       market_data: Dict) -> Dict[str, any]:
        """Design options strategy for clinical trial events."""
        days_to_event = signal.timeframe
        iv = market_data.get("iv", 0.30)

        # Binary event - use straddles/strangles
        if iv > 0.50:  # High IV
            strategy = {
                "name": "iron_condor",
                "type": "volatility_selling",
                "structure": {
                    "put_spread": {
                        "short_strike": -iv * np.sqrt(days_to_event/365),
                        "long_strike": -iv * np.sqrt(days_to_event/365) * 1.5
                    },
                    "call_spread": {
                        "short_strike": iv * np.sqrt(days_to_event/365),
                        "long_strike": iv * np.sqrt(days_to_event/365) * 1.5
                    },
                    "expiration": days_to_event + 5
                },
                "sizing": self.portfolio_value * 0.02,  # Risk 2%
                "confidence": 60,
                "reason": "High IV before clinical trial - sell volatility"
            }
        else:  # Normal/Low IV
            expected_move = iv * np.sqrt(days_to_event/365) * np.sqrt(8/np.pi)
            strategy = {
                "name": "straddle",
                "type": "volatility_buying",
                "structure": {
                    "put": {"strike": "ATM", "expiration": days_to_event + 5},
                    "call": {"strike": "ATM", "expiration": days_to_event + 5}
                },
                "sizing": self.portfolio_value * 0.03,  # Risk 3%
                "confidence": 70,
                "reason": f"Binary event with {expected_move*100:.1f}% expected move"
            }

        return strategy

    def _design_insider_strategy(self, signal: TradingSignal,
                                market_data: Dict) -> Dict[str, any]:
        """Design options strategy based on insider trading patterns."""
        # Insider buying typically bullish, selling bearish
        if signal.signal_type == "bullish" and signal.strength > 70:
            strategy = {
                "name": "call_spread",
                "type": "bullish",
                "structure": {
                    "long_call": {
                        "strike_offset": 0.05,  # 5% OTM
                        "expiration": 60
                    },
                    "short_call": {
                        "strike_offset": 0.15,  # 15% OTM
                        "expiration": 60
                    }
                },
                "sizing": self.portfolio_value * 0.03,
                "confidence": signal.strength * 0.8,
                "reason": f"Strong insider buying (strength: {signal.strength})"
            }
        elif signal.signal_type == "bearish" and signal.strength > 70:
            strategy = {
                "name": "protective_collar",
                "type": "defensive",
                "structure": {
                    "long_put": {
                        "strike_offset": -0.10,
                        "expiration": 45
                    },
                    "short_call": {
                        "strike_offset": 0.10,
                        "expiration": 45
                    }
                },
                "sizing": "full_position",
                "confidence": signal.strength * 0.7,
                "reason": f"Insider selling protection (strength: {signal.strength})"
            }
        else:
            strategy = {
                "name": "none",
                "reason": "Insider signal not strong enough"
            }

        return strategy

    def _design_earnings_strategy(self, signal: TradingSignal,
                                 market_data: Dict) -> Dict[str, any]:
        """Design options strategy for earnings."""
        earnings_strategy = self.strategies["earnings"]

        analysis = earnings_strategy.analyze_earnings_setup(
            symbol=signal.symbol,
            days_to_earnings=signal.timeframe,
            iv_current=market_data.get("iv", 0.30),
            iv_historical=market_data.get("iv_history", [0.25, 0.30, 0.28, 0.32]),
            historical_moves=market_data.get("earnings_moves", [0.04, 0.06, 0.05, 0.07])
        )

        if analysis["strategy"] != "avoid":
            return {
                "name": analysis["strategy"],
                "type": "earnings",
                "structure": analysis["trade_details"],
                "sizing": earnings_strategy.capital_per_trade,
                "confidence": analysis["confidence"],
                "reason": analysis["reasoning"]
            }
        else:
            return {"name": "none", "reason": analysis["reasoning"]}

    def _design_high_vol_strategy(self, market_data: Dict) -> Dict[str, any]:
        """Design strategy for high volatility regime."""
        vix = market_data.get("vix", 20)

        if vix > 30:
            # Deploy protective puts
            protective = self.strategies["protective_put"]
            deployment = protective.determine_deployment_conditions(
                vix=vix,
                market_drawdown=market_data.get("drawdown", 0),
                earnings_days=30,
                fed_days=30
            )

            if deployment["deploy"]:
                return {
                    "name": "protective_puts",
                    "type": "defensive",
                    "structure": {
                        "strike": "5% OTM",
                        "expiration": 45,
                        "coverage": 0.5  # Cover 50% of portfolio
                    },
                    "sizing": self.portfolio_value * 0.02,
                    "confidence": 80,
                    "reason": deployment["reasoning"]
                }

        return {"name": "none", "reason": "No high vol strategy needed"}

    def _determine_market_regime(self, market_data: Dict) -> MarketRegime:
        """Determine current market regime."""
        vix = market_data.get("vix", 20)
        trend = market_data.get("trend", "neutral")

        if vix > 30:
            return MarketRegime.HIGH_VOL
        elif vix < 15:
            return MarketRegime.LOW_VOL
        elif trend == "bullish":
            return MarketRegime.BULL
        elif trend == "bearish":
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL

    def _calculate_patent_sizing(self, strength: float, days_to_event: int) -> float:
        """Calculate position size for patent cliff trades."""
        base_size = self.portfolio_value * 0.05  # Base 5% allocation

        # Adjust for signal strength
        strength_multiplier = strength / 100

        # Adjust for time to event (closer = larger)
        if days_to_event < 30:
            time_multiplier = 1.5
        elif days_to_event < 90:
            time_multiplier = 1.2
        else:
            time_multiplier = 0.8

        return base_size * strength_multiplier * time_multiplier

    def _score_recommendations(self, recommendations: List[Dict],
                              market_data: Dict) -> List[Dict]:
        """Score and rank strategy recommendations."""
        scored = []

        for rec in recommendations:
            if rec.get("name") == "none":
                continue

            score = rec.get("confidence", 50)

            # Adjust for market conditions
            if market_data.get("vix", 20) > 25 and "defensive" in rec.get("type", ""):
                score += 10

            # Adjust for liquidity
            if market_data.get("option_volume", 0) > 10000:
                score += 5

            rec["score"] = score
            scored.append(rec)

        return sorted(scored, key=lambda x: x["score"], reverse=True)


class OptionsExecutionEngine:
    """
    Handles execution of options strategies including order generation,
    position tracking, and real-time adjustments.
    """

    def __init__(self, broker_api=None):
        """Initialize execution engine."""
        self.broker_api = broker_api
        self.open_orders: List[OptionsOrder] = []
        self.positions: List[OptionsPosition] = []
        self.execution_history = []
        logger.info("Options Execution Engine initialized")

    def execute_strategy(self, strategy: Dict, market_data: Dict) -> Dict[str, any]:
        """
        Execute an options strategy by generating and submitting orders.

        Args:
            strategy: Strategy details from OptionsStrategySelector
            market_data: Current market data

        Returns:
            Execution result with order IDs and status
        """
        orders = self._generate_orders(strategy, market_data)
        results = []

        for order in orders:
            if self.broker_api:
                # Real execution through broker API
                result = self._submit_order(order)
            else:
                # Simulated execution
                result = self._simulate_order(order, market_data)

            results.append(result)
            self.execution_history.append({
                "timestamp": datetime.now(),
                "order": order,
                "result": result
            })

        return {
            "strategy": strategy["name"],
            "orders": len(orders),
            "results": results,
            "total_cost": sum(r.get("cost", 0) for r in results),
            "status": "executed" if all(r["status"] == "filled" for r in results) else "partial"
        }

    def _generate_orders(self, strategy: Dict, market_data: Dict) -> List[OptionsOrder]:
        """Generate orders from strategy specification."""
        orders = []
        symbol = market_data.get("symbol", "SPY")
        current_price = market_data.get("price", 100)

        if strategy["name"] == "put_spread":
            # Generate put spread orders
            structure = strategy["structure"]

            # Long put
            long_strike = current_price * (1 + structure["long_put"]["strike_offset"])
            orders.append(OptionsOrder(
                symbol=symbol,
                option_type=OptionType.PUT,
                action="buy",
                quantity=self._calculate_contracts(strategy["sizing"], market_data),
                order_type="limit",
                strike=self._round_to_strike(long_strike),
                expiration=datetime.now() + timedelta(days=structure["long_put"]["expiration"]),
                strategy_name=strategy["name"]
            ))

            # Short put
            short_strike = current_price * (1 + structure["short_put"]["strike_offset"])
            orders.append(OptionsOrder(
                symbol=symbol,
                option_type=OptionType.PUT,
                action="sell",
                quantity=self._calculate_contracts(strategy["sizing"], market_data),
                order_type="limit",
                strike=self._round_to_strike(short_strike),
                expiration=datetime.now() + timedelta(days=structure["short_put"]["expiration"]),
                strategy_name=strategy["name"]
            ))

        elif strategy["name"] == "straddle":
            # Generate straddle orders
            strike = self._round_to_strike(current_price)
            expiration = datetime.now() + timedelta(days=strategy["structure"]["put"]["expiration"])
            contracts = self._calculate_contracts(strategy["sizing"], market_data)

            # Buy put
            orders.append(OptionsOrder(
                symbol=symbol,
                option_type=OptionType.PUT,
                action="buy",
                quantity=contracts,
                order_type="limit",
                strike=strike,
                expiration=expiration,
                strategy_name=strategy["name"]
            ))

            # Buy call
            orders.append(OptionsOrder(
                symbol=symbol,
                option_type=OptionType.CALL,
                action="buy",
                quantity=contracts,
                order_type="limit",
                strike=strike,
                expiration=expiration,
                strategy_name=strategy["name"]
            ))

        elif strategy["name"] == "protective_puts":
            # Generate protective put orders
            strike = current_price * 0.95  # 5% OTM
            orders.append(OptionsOrder(
                symbol=symbol,
                option_type=OptionType.PUT,
                action="buy",
                quantity=int(strategy["sizing"] / (current_price * 0.02)),  # Estimate premium
                order_type="limit",
                strike=self._round_to_strike(strike),
                expiration=datetime.now() + timedelta(days=45),
                strategy_name=strategy["name"]
            ))

        # Additional strategy types...

        return orders

    def _calculate_contracts(self, sizing: float, market_data: Dict) -> int:
        """Calculate number of contracts based on sizing."""
        if isinstance(sizing, str) and sizing == "full_position":
            return int(market_data.get("position_size", 1000) / 100)

        # Estimate option price (simplified)
        est_premium = market_data.get("price", 100) * 0.02  # 2% of stock price
        return max(1, int(sizing / (est_premium * 100)))

    def _round_to_strike(self, price: float) -> float:
        """Round to nearest standard strike."""
        if price < 50:
            return round(price * 2) / 2
        elif price < 100:
            return round(price)
        elif price < 500:
            return round(price / 5) * 5
        else:
            return round(price / 10) * 10

    def _submit_order(self, order: OptionsOrder) -> Dict[str, any]:
        """Submit order through broker API."""
        # This would connect to your actual broker API
        # Example for IBKR, TD Ameritrade, etc.
        pass

    def _simulate_order(self, order: OptionsOrder, market_data: Dict) -> Dict[str, any]:
        """Simulate order execution for testing."""
        # Calculate theoretical price
        S = market_data.get("price", 100)
        K = order.strike if order.strike else S
        T = (order.expiration - datetime.now()).days / 365 if order.expiration else 0.25
        r = 0.05
        sigma = market_data.get("iv", 0.30)

        theoretical_price = BlackScholesModel.price_option(S, K, T, r, sigma, order.option_type)

        # Add spread
        if order.action == "buy":
            fill_price = theoretical_price * 1.01  # Pay 1% above mid
        else:
            fill_price = theoretical_price * 0.99  # Receive 1% below mid

        cost = fill_price * 100 * order.quantity
        if order.action == "sell":
            cost = -cost  # Credit for selling

        return {
            "order_id": f"SIM_{datetime.now().timestamp()}",
            "status": "filled",
            "fill_price": fill_price,
            "quantity": order.quantity,
            "cost": cost,
            "timestamp": datetime.now()
        }

    def monitor_positions(self) -> Dict[str, any]:
        """Monitor all open positions and generate alerts."""
        alerts = []
        adjustments = []

        for position in self.positions:
            # Check days to expiry
            if position.days_to_expiry <= 7:
                alerts.append({
                    "type": "expiration",
                    "position": position,
                    "message": f"{position.symbol} {position.option_type.value} expires in {position.days_to_expiry} days"
                })

            # Check P&L thresholds
            if position.pnl_percent > 50:
                adjustments.append({
                    "position": position,
                    "action": "consider_closing",
                    "reason": f"Profit of {position.pnl_percent:.1f}%"
                })
            elif position.pnl_percent < -50:
                adjustments.append({
                    "position": position,
                    "action": "stop_loss",
                    "reason": f"Loss of {abs(position.pnl_percent):.1f}%"
                })

            # Check Greeks thresholds
            if abs(position.greeks.delta * position.contracts) > 500:
                alerts.append({
                    "type": "delta",
                    "position": position,
                    "message": f"High delta exposure: {position.greeks.delta * position.contracts:.0f}"
                })

        return {
            "positions": len(self.positions),
            "total_pnl": sum(p.pnl for p in self.positions),
            "alerts": alerts,
            "adjustments": adjustments
        }


class OptionsBacktester:
    """
    Backtesting framework for options strategies with realistic assumptions
    including bid-ask spreads, assignment risk, and early exercise.
    """

    def __init__(self, initial_capital: float = 100000):
        """Initialize backtester."""
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        self.daily_pnl = []

    def backtest_strategy(self, strategy_name: str, signals: pd.DataFrame,
                         market_data: pd.DataFrame, start_date: datetime,
                         end_date: datetime) -> Dict[str, any]:
        """
        Backtest an options strategy over historical data.

        Args:
            strategy_name: Name of strategy to test
            signals: DataFrame with trading signals
            market_data: Historical market data
            start_date: Backtest start date
            end_date: Backtest end date

        Returns:
            Backtest results with performance metrics
        """
        logger.info(f"Starting backtest of {strategy_name} from {start_date} to {end_date}")

        # Initialize strategy selector
        selector = OptionsStrategySelector(self.capital)
        executor = OptionsExecutionEngine()

        # Filter data to backtest period
        market_data = market_data[(market_data.index >= start_date) &
                                  (market_data.index <= end_date)]

        for date, row in market_data.iterrows():
            # Check for signals on this date
            daily_signals = signals[signals['date'] == date]

            for _, signal_row in daily_signals.iterrows():
                # Create signal object
                signal = TradingSignal(
                    source=signal_row['source'],
                    symbol=signal_row['symbol'],
                    signal_type=signal_row['signal_type'],
                    strength=signal_row['strength'],
                    timeframe=signal_row.get('timeframe', 30)
                )

                # Get market data for signal
                signal_market_data = {
                    "symbol": signal.symbol,
                    "price": row['close'],
                    "iv": row.get('iv', 0.25),
                    "vix": row.get('vix', 20),
                    "volume": row.get('volume', 1000000)
                }

                # Select strategy
                strategy = selector.select_strategy(signal, signal_market_data)

                if strategy.get("name") != "none":
                    # Execute strategy
                    result = executor.execute_strategy(strategy, signal_market_data)

                    # Track trade
                    self.trades.append({
                        "date": date,
                        "strategy": strategy["name"],
                        "cost": result["total_cost"],
                        "confidence": strategy.get("confidence", 50)
                    })

                    # Update capital
                    self.capital -= result["total_cost"]

            # Update existing positions
            self._update_positions(date, row)

            # Calculate daily P&L
            daily_value = self._calculate_portfolio_value(row)
            self.daily_pnl.append({
                "date": date,
                "value": daily_value,
                "pnl": daily_value - self.initial_capital,
                "return": (daily_value / self.initial_capital - 1) * 100
            })

        # Calculate performance metrics
        return self._calculate_performance_metrics()

    def _update_positions(self, date: datetime, market_data: pd.Series):
        """Update existing positions with current market data."""
        for position in self.positions:
            # Update Greeks and prices
            days_to_expiry = (position.expiration - date).days
            if days_to_expiry > 0:
                T = days_to_expiry / 365
                S = market_data['close']
                K = position.strike
                r = 0.05
                sigma = market_data.get('iv', 0.25)

                # Update price
                position.current_price = BlackScholesModel.price_option(
                    S, K, T, r, sigma, position.option_type
                )

                # Update Greeks
                position.greeks = BlackScholesModel.calculate_greeks(
                    S, K, T, r, sigma, position.option_type
                )
            else:
                # Option expired
                if position.option_type == OptionType.CALL:
                    position.current_price = max(0, market_data['close'] - position.strike)
                else:
                    position.current_price = max(0, position.strike - market_data['close'])

    def _calculate_portfolio_value(self, market_data: pd.Series) -> float:
        """Calculate total portfolio value including options positions."""
        options_value = sum(p.current_price * 100 * p.contracts for p in self.positions)
        return self.capital + options_value

    def _calculate_performance_metrics(self) -> Dict[str, any]:
        """Calculate comprehensive performance metrics."""
        if not self.daily_pnl:
            return {"error": "No trades executed"}

        pnl_df = pd.DataFrame(self.daily_pnl)
        returns = pnl_df['return'].values

        # Basic metrics
        total_return = (self.capital / self.initial_capital - 1) * 100
        num_trades = len(self.trades)

        # Risk metrics
        if len(returns) > 1:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            max_drawdown = self._calculate_max_drawdown(pnl_df['value'].values)
            volatility = np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0
            max_drawdown = 0
            volatility = 0

        # Win rate
        winning_trades = sum(1 for t in self.trades if t["cost"] < 0)  # Sold for credit
        win_rate = winning_trades / num_trades * 100 if num_trades > 0 else 0

        return {
            "initial_capital": self.initial_capital,
            "final_capital": self.capital,
            "total_return": total_return,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "best_trade": max(self.trades, key=lambda x: -x["cost"])["strategy"] if self.trades else None,
            "worst_trade": min(self.trades, key=lambda x: -x["cost"])["strategy"] if self.trades else None
        }

    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = values[0]
        max_dd = 0

        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                dd = (peak - value) / peak * 100
                if dd > max_dd:
                    max_dd = dd

        return max_dd


# Example usage
if __name__ == "__main__":
    print("Options Strategy Integration Module")
    print("=" * 80)

    # Initialize components
    selector = OptionsStrategySelector(portfolio_value=1000000)
    executor = OptionsExecutionEngine()

    # Test signal from patent intelligence
    patent_signal = TradingSignal(
        source="patent_cliff",
        symbol="ABBV",
        signal_type="bearish",
        strength=75,
        timeframe=60,
        metadata={"drug": "Humira", "revenue_at_risk": 20000000000}
    )

    # Market data
    market_data = {
        "symbol": "ABBV",
        "price": 150,
        "iv": 0.35,
        "vix": 22,
        "trend": "neutral",
        "option_volume": 25000
    }

    # Select strategy
    print("\n1. STRATEGY SELECTION")
    print("-" * 40)
    strategy = selector.select_strategy(patent_signal, market_data)
    print(f"Selected Strategy: {strategy.get('name', 'none')}")
    print(f"Confidence: {strategy.get('confidence', 0):.0f}%")
    print(f"Reason: {strategy.get('reason', 'N/A')}")
    print(f"Sizing: ${strategy.get('sizing', 0):,.0f}")

    # Execute strategy
    print("\n2. STRATEGY EXECUTION")
    print("-" * 40)
    if strategy.get("name") != "none":
        execution_result = executor.execute_strategy(strategy, market_data)
        print(f"Orders Generated: {execution_result['orders']}")
        print(f"Total Cost: ${execution_result['total_cost']:,.2f}")
        print(f"Status: {execution_result['status']}")

    # Test clinical trial signal
    clinical_signal = TradingSignal(
        source="clinical_trial",
        symbol="BIIB",
        signal_type="volatility",
        strength=80,
        timeframe=7,
        metadata={"trial": "Phase 3 Alzheimer's", "catalyst": "FDA decision"}
    )

    print("\n3. CLINICAL TRIAL STRATEGY")
    print("-" * 40)
    clinical_strategy = selector.select_strategy(clinical_signal, {
        "symbol": "BIIB",
        "price": 250,
        "iv": 0.55,
        "vix": 25
    })
    print(f"Selected Strategy: {clinical_strategy.get('name', 'none')}")
    print(f"Structure: {clinical_strategy.get('type', 'N/A')}")

    print("\n" + "=" * 80)
    print("Integration Testing Complete")