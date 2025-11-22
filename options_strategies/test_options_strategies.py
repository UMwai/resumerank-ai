"""
Comprehensive Test Suite for Options Strategies

Tests all options strategies with various market conditions,
validates risk parameters, and ensures proper integration.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from options_playbook import (
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

from options_integration import (
    OptionsStrategySelector,
    OptionsExecutionEngine,
    OptionsBacktester,
    TradingSignal
)


class TestProtectivePutStrategy:
    """Test suite for Protective Put Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a protective put strategy instance."""
        return ProtectivePutStrategy(portfolio_value=1000000, max_insurance_cost=0.02)

    def test_deployment_conditions_high_vix(self, strategy):
        """Test deployment triggers with high VIX."""
        result = strategy.determine_deployment_conditions(
            vix=35,
            market_drawdown=3,
            earnings_days=30,
            fed_days=30
        )

        assert result["deploy"] is True
        assert result["urgency"] == "high"
        assert result["sizing_multiplier"] > 1.0
        assert "VIX elevated" in result["reasoning"]

    def test_deployment_conditions_drawdown(self, strategy):
        """Test deployment triggers with market drawdown."""
        result = strategy.determine_deployment_conditions(
            vix=18,
            market_drawdown=12,
            earnings_days=30,
            fed_days=30
        )

        assert result["deploy"] is True
        assert result["urgency"] == "high"
        assert "drawdown" in result["reasoning"].lower()

    def test_strike_selection_standard(self, strategy):
        """Test standard strike selection."""
        result = strategy.select_strike(
            current_price=100,
            volatility=0.25,
            protection_level="standard"
        )

        assert result["strike"] == 95.0  # 5% OTM
        assert result["moneyness"] == pytest.approx(-5.0, rel=0.1)
        assert result["max_loss_percent"] == 5.0

    def test_strike_selection_high_volatility(self, strategy):
        """Test strike selection in high volatility."""
        result = strategy.select_strike(
            current_price=100,
            volatility=0.40,
            protection_level="standard"
        )

        # Should be closer to ATM in high vol
        assert result["strike"] > 95.0
        assert result["strike"] <= 97.5

    def test_expiration_selection(self, strategy):
        """Test expiration date selection."""
        event_calendar = {
            "earnings": 20,
            "fed_meeting": 35,
            "economic_data": 10
        }

        result = strategy.select_expiration(
            volatility_regime="normal",
            event_calendar=event_calendar
        )

        assert result["target_dte"] >= 45
        assert "fed_meeting" in result["covers_events"]
        assert result["theta_decay_acceleration"] is False

    def test_position_sizing(self, strategy):
        """Test position sizing calculation."""
        result = strategy.calculate_position_size(
            portfolio_value=1000000,
            put_price=2.50,
            volatility=0.25,
            correlation=0.85
        )

        assert result["num_contracts"] > 0
        assert result["cost_as_percent"] <= 2.0  # Within max insurance cost
        assert result["effective_protection"] > 0

    def test_rolling_strategy_near_expiry(self, strategy):
        """Test rolling decision near expiration."""
        result = strategy.generate_rolling_strategy(
            current_dte=10,
            pnl_percent=-30,
            underlying_move=2
        )

        assert result["action"] == "roll"
        assert "Approaching expiration" in result["reasoning"]
        assert result["new_dte_target"] == 45


class TestCoveredCallStrategy:
    """Test suite for Covered Call Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a covered call strategy instance."""
        return CoveredCallStrategy(position_size=1000, target_monthly_income=0.02)

    def test_strike_selection_high_iv_rank(self, strategy):
        """Test strike selection with high IV rank."""
        result = strategy.select_strike(
            current_price=100,
            volatility=0.30,
            technical_resistance=105,
            iv_rank=75
        )

        assert result["strike"] >= 102  # At least 2% OTM
        assert result["strike"] <= 105  # Respect resistance
        assert result["annualized_return"] > 0
        assert result["assignment_probability"] < 50  # Should be OTM

    def test_expiration_cycle_selection(self, strategy):
        """Test expiration cycle selection."""
        earnings_date = datetime.now() + timedelta(days=20)

        result = strategy.select_expiration_cycle(
            volatility=0.35,
            earnings_date=earnings_date,
            dividend_date=None
        )

        assert result["cycle"] == "weekly"  # High vol = short cycle
        assert result["target_dte"] < 20  # Before earnings
        assert "before earnings" in result["reasoning"]

    def test_holdings_selection(self, strategy):
        """Test which holdings to write calls against."""
        holdings = [
            {
                "symbol": "AAPL",
                "iv_rank": 65,
                "volatility": 0.28,
                "trend": "neutral",
                "option_volume": 50000,
                "dividend_yield": 0.015
            },
            {
                "symbol": "MSFT",
                "iv_rank": 45,
                "volatility": 0.22,
                "trend": "mild_bullish",
                "option_volume": 30000,
                "dividend_yield": 0.025
            },
            {
                "symbol": "TSLA",
                "iv_rank": 80,
                "volatility": 0.55,  # Too high
                "trend": "bearish",
                "option_volume": 100000,
                "dividend_yield": 0
            }
        ]

        result = strategy.determine_holdings_to_write(holdings, iv_threshold=50)

        assert len(result) >= 1
        assert result[0]["symbol"] == "AAPL"  # Best candidate
        assert result[0]["score"] > 50

    def test_assignment_risk_management_itm(self, strategy):
        """Test assignment risk management for ITM calls."""
        result = strategy.manage_assignment_risk(
            dte=10,
            moneyness=-2,  # 2% ITM
            delta=-0.75,
            pnl=-150
        )

        assert result["action"] in ["roll_up_and_out", "evaluate_rolling"]
        assert result["urgency"] == "normal"
        assert result["estimated_assignment_prob"] > 70


class TestVolatilityArbitrageStrategy:
    """Test suite for Volatility Arbitrage Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a volatility arbitrage strategy instance."""
        return VolatilityArbitrageStrategy(capital_allocated=100000, max_vega_exposure=10000)

    def test_iv_rv_divergence_sell_signal(self, strategy):
        """Test IV/RV divergence identification for selling vol."""
        result = strategy.identify_iv_rv_divergence(
            symbol="SPY",
            iv_30d=0.30,
            rv_30d=0.20,
            rv_60d=0.22
        )

        assert result["signal"] == "sell_volatility"
        assert result["confidence"] > 50
        assert result["strategy"] in ["short_straddle", "iron_condor"]
        assert result["metrics"]["iv_rv_spread"] == pytest.approx(0.10, rel=0.01)

    def test_iv_rv_divergence_buy_signal(self, strategy):
        """Test IV/RV divergence for buying vol."""
        result = strategy.identify_iv_rv_divergence(
            symbol="SPY",
            iv_30d=0.15,
            rv_30d=0.22,
            rv_60d=0.20
        )

        assert result["signal"] == "buy_volatility"
        assert result["strategy"] in ["long_straddle", "long_strangle"]

    def test_vix_futures_contango(self, strategy):
        """Test VIX futures analysis in contango."""
        result = strategy.analyze_vix_futures(
            spot_vix=15,
            front_month=16.5,
            second_month=18,
            days_to_expiry=20
        )

        assert result["structure"] == "contango"
        assert result["signal"] in ["short_vix_futures", "mild_contango"]
        assert result["metrics"]["roll_yield"] > 0

    def test_vix_futures_backwardation(self, strategy):
        """Test VIX futures analysis in backwardation."""
        result = strategy.analyze_vix_futures(
            spot_vix=30,
            front_month=28,
            second_month=26,
            days_to_expiry=15
        )

        assert result["structure"] == "backwardation"
        assert result["signal"] in ["long_vix_futures", "mild_backwardation"]

    def test_calendar_spread_design(self, strategy):
        """Test calendar spread design."""
        iv_term_structure = {
            30: 0.20,
            60: 0.28,
            90: 0.30
        }

        result = strategy.design_calendar_spread(
            symbol="SPY",
            iv_term_structure=iv_term_structure,
            current_price=400
        )

        if result["signal"] != "no_opportunity":
            assert "edge" in result
            assert result["edge"] != 0
            assert result["num_spreads"] > 0
            assert "breakeven_points" in result

    def test_position_limits(self, strategy):
        """Test position limit calculations."""
        result = strategy.calculate_position_limits(
            portfolio_vega=5000,
            portfolio_gamma=100
        )

        assert result["vega_remaining"] == 5000  # 10000 - 5000
        assert result["vega_contracts"] >= 0
        assert result["max_premium"] == 10000  # 10% of capital
        assert result["suggested_trades"] >= 0


class TestEarningsStrategy:
    """Test suite for Earnings Strategy."""

    @pytest.fixture
    def strategy(self):
        """Create an earnings strategy instance."""
        return EarningsStrategy(capital_per_trade=5000)

    def test_earnings_setup_high_iv(self, strategy):
        """Test earnings setup analysis with high IV."""
        result = strategy.analyze_earnings_setup(
            symbol="NFLX",
            days_to_earnings=3,
            iv_current=0.60,
            iv_historical=[0.35, 0.40, 0.38, 0.42],
            historical_moves=[0.08, 0.10, 0.07, 0.12]
        )

        assert result["strategy"] in ["short_strangle", "iron_condor"]
        assert result["confidence"] >= 60
        assert "elevated" in result["reasoning"].lower()

    def test_earnings_setup_low_iv(self, strategy):
        """Test earnings setup with low IV."""
        result = strategy.analyze_earnings_setup(
            symbol="AAPL",
            days_to_earnings=5,
            iv_current=0.25,
            iv_historical=[0.35, 0.40, 0.38, 0.42],
            historical_moves=[0.06, 0.08, 0.07, 0.09]
        )

        assert result["strategy"] in ["long_straddle", "avoid"]
        if result["strategy"] == "long_straddle":
            assert "exceed" in result["reasoning"].lower()

    def test_straddle_design(self, strategy):
        """Test straddle position design."""
        result = strategy.design_straddle_strangle(
            stock_price=100,
            iv=0.40,
            expected_move=0.08,
            capital=5000,
            use_straddle=True
        )

        assert result["position_type"] == "straddle"
        assert result["legs"]["put"]["strike"] == result["legs"]["call"]["strike"]
        assert result["cost"] <= 5000
        assert result["breakevens"]["required_move_pct"] > 0

    def test_strangle_design(self, strategy):
        """Test strangle position design."""
        result = strategy.design_straddle_strangle(
            stock_price=100,
            iv=0.35,
            expected_move=0.06,
            capital=5000,
            use_straddle=False
        )

        assert result["position_type"] == "strangle"
        assert result["legs"]["put"]["strike"] < result["legs"]["call"]["strike"]
        assert result["cost"] <= 5000

    def test_entry_timing(self, strategy):
        """Test entry timing determination."""
        iv_curve = {
            10: 0.30,
            7: 0.32,
            5: 0.38,
            3: 0.45,
            1: 0.55
        }

        result = strategy.determine_entry_timing(
            days_to_earnings=10,
            iv_curve=iv_curve,
            historical_iv_pattern="late_spike"
        )

        assert result["ideal_entry_dte"] <= 7
        assert "spike" in result["reasoning"].lower()

    def test_position_sizing_high_confidence(self, strategy):
        """Test position sizing with high confidence."""
        result = strategy.calculate_position_size(
            confidence=85,
            kelly_fraction=0.15,
            max_loss=1000
        )

        assert result["recommended_size"] > strategy.capital_per_trade
        assert result["recommended_size"] <= strategy.capital_per_trade * 2

    def test_exit_plan_long_straddle(self, strategy):
        """Test exit plan for long straddle."""
        result = strategy.generate_exit_plan(
            strategy="long_straddle",
            entry_price=10,
            expected_move=0.08,
            days_to_earnings=5
        )

        assert "before_earnings" in result
        assert "after_earnings" in result
        assert result["stop_loss"] == 5.0  # 50% stop
        assert result["time_stop"] == 6  # Day after earnings


class TestOptionsIntegration:
    """Test suite for options integration components."""

    @pytest.fixture
    def selector(self):
        """Create strategy selector instance."""
        return OptionsStrategySelector(portfolio_value=1000000)

    @pytest.fixture
    def executor(self):
        """Create execution engine instance."""
        return OptionsExecutionEngine()

    def test_patent_cliff_strategy_selection(self, selector):
        """Test strategy selection for patent cliff signal."""
        signal = TradingSignal(
            source="patent_cliff",
            symbol="ABBV",
            signal_type="bearish",
            strength=80,
            timeframe=60
        )

        market_data = {
            "symbol": "ABBV",
            "price": 150,
            "iv": 0.35,
            "vix": 22,
            "option_volume": 20000
        }

        result = selector.select_strategy(signal, market_data)

        assert result["name"] == "put_spread"
        assert result["type"] == "bearish"
        assert result["confidence"] > 50

    def test_clinical_trial_strategy_selection(self, selector):
        """Test strategy selection for clinical trial signal."""
        signal = TradingSignal(
            source="clinical_trial",
            symbol="BIIB",
            signal_type="volatility",
            strength=75,
            timeframe=7
        )

        market_data = {
            "symbol": "BIIB",
            "price": 250,
            "iv": 0.55,
            "vix": 25
        }

        result = selector.select_strategy(signal, market_data)

        assert result["name"] in ["iron_condor", "straddle"]
        assert "volatility" in result["type"]

    def test_order_generation_put_spread(self, executor):
        """Test order generation for put spread."""
        strategy = {
            "name": "put_spread",
            "structure": {
                "long_put": {"strike_offset": -0.05, "expiration": 30},
                "short_put": {"strike_offset": -0.10, "expiration": 30}
            },
            "sizing": 10000
        }

        market_data = {"symbol": "SPY", "price": 400}

        orders = executor._generate_orders(strategy, market_data)

        assert len(orders) == 2
        assert orders[0].action == "buy"
        assert orders[1].action == "sell"
        assert orders[0].strike > orders[1].strike

    def test_simulated_execution(self, executor):
        """Test simulated order execution."""
        order = OptionsOrder(
            symbol="SPY",
            option_type=OptionType.PUT,
            action="buy",
            quantity=10,
            order_type="limit",
            strike=400,
            expiration=datetime.now() + timedelta(days=30)
        )

        market_data = {"price": 410, "iv": 0.20}

        result = executor._simulate_order(order, market_data)

        assert result["status"] == "filled"
        assert result["fill_price"] > 0
        assert result["cost"] > 0  # Buying costs money


class TestBlackScholesModel:
    """Test suite for Black-Scholes calculations."""

    def test_call_option_pricing(self):
        """Test call option pricing."""
        price = BlackScholesModel.price_option(
            S=100, K=105, T=0.25, r=0.05, sigma=0.20,
            option_type=OptionType.CALL
        )

        assert price > 0
        assert price < 100  # Can't be worth more than stock

    def test_put_option_pricing(self):
        """Test put option pricing."""
        price = BlackScholesModel.price_option(
            S=100, K=105, T=0.25, r=0.05, sigma=0.20,
            option_type=OptionType.PUT
        )

        assert price > 5  # Has intrinsic value
        assert price < 105  # Can't be worth more than strike

    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        greeks = BlackScholesModel.calculate_greeks(
            S=100, K=100, T=0.25, r=0.05, sigma=0.25,
            option_type=OptionType.CALL
        )

        assert 0 < greeks.delta < 1  # Call delta between 0 and 1
        assert greeks.gamma > 0  # Gamma always positive
        assert greeks.vega > 0  # Vega positive for long options
        assert greeks.theta < 0  # Theta negative for long options

    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        S, K, T, r, sigma = 100, 100, 1, 0.05, 0.25

        call_price = BlackScholesModel.price_option(
            S, K, T, r, sigma, OptionType.CALL
        )
        put_price = BlackScholesModel.price_option(
            S, K, T, r, sigma, OptionType.PUT
        )

        # Put-Call Parity: C - P = S - K*exp(-rT)
        parity = call_price - put_price
        expected = S - K * np.exp(-r * T)

        assert parity == pytest.approx(expected, rel=0.01)


class TestRiskManager:
    """Test suite for options risk management."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return OptionsRiskManager(portfolio_value=1000000, max_portfolio_theta=-1000)

    def test_portfolio_greeks_calculation(self, risk_manager):
        """Test portfolio Greeks aggregation."""
        # Add test positions
        position1 = OptionsPosition(
            symbol="SPY",
            option_type=OptionType.CALL,
            strike=400,
            expiration=datetime.now() + timedelta(days=30),
            contracts=10,
            entry_price=5,
            current_price=6,
            underlying_price=405,
            implied_volatility=0.20,
            greeks=OptionsGreeks(delta=0.5, gamma=0.02, vega=0.3, theta=-0.5, rho=0.1)
        )

        position2 = OptionsPosition(
            symbol="SPY",
            option_type=OptionType.PUT,
            strike=395,
            expiration=datetime.now() + timedelta(days=30),
            contracts=10,
            entry_price=4,
            current_price=3,
            underlying_price=405,
            implied_volatility=0.20,
            greeks=OptionsGreeks(delta=-0.3, gamma=0.02, vega=0.3, theta=-0.4, rho=-0.05)
        )

        risk_manager.positions = [position1, position2]

        greeks = risk_manager.calculate_portfolio_greeks()

        assert greeks["delta"] == pytest.approx(2.0, rel=0.01)  # (0.5 - 0.3) * 10
        assert greeks["gamma"] == pytest.approx(0.4, rel=0.01)  # (0.02 + 0.02) * 10
        assert greeks["theta"] == pytest.approx(-9.0, rel=0.01)  # (-0.5 - 0.4) * 10

    def test_risk_limit_check_pass(self, risk_manager):
        """Test risk limit check that passes."""
        new_position = OptionsPosition(
            symbol="QQQ",
            option_type=OptionType.CALL,
            strike=350,
            expiration=datetime.now() + timedelta(days=30),
            contracts=5,
            entry_price=3,
            current_price=3,
            underlying_price=352,
            implied_volatility=0.22,
            greeks=OptionsGreeks(delta=0.4, gamma=0.01, vega=0.2, theta=-0.3, rho=0.05)
        )

        result = risk_manager.check_risk_limits(new_position)

        assert result["approved"] is True
        assert len(result["violations"]) == 0

    def test_risk_limit_check_theta_breach(self, risk_manager):
        """Test risk limit check with theta breach."""
        new_position = OptionsPosition(
            symbol="SPY",
            option_type=OptionType.CALL,
            strike=400,
            expiration=datetime.now() + timedelta(days=5),  # Near expiry
            contracts=100,  # Large position
            entry_price=2,
            current_price=2,
            underlying_price=401,
            implied_volatility=0.30,
            greeks=OptionsGreeks(delta=0.6, gamma=0.05, vega=0.1, theta=-15, rho=0.01)
        )

        result = risk_manager.check_risk_limits(new_position)

        assert result["approved"] is False
        assert len(result["violations"]) > 0
        assert any("Theta" in v for v in result["violations"])


# Performance and stress tests
class TestPerformanceAndStress:
    """Performance and stress tests for options strategies."""

    def test_large_portfolio_processing(self):
        """Test processing large number of positions."""
        risk_manager = OptionsRiskManager(portfolio_value=10000000)

        # Create 100 random positions
        for i in range(100):
            position = OptionsPosition(
                symbol=f"TEST{i}",
                option_type=OptionType.CALL if i % 2 == 0 else OptionType.PUT,
                strike=100 + i,
                expiration=datetime.now() + timedelta(days=30 + i),
                contracts=np.random.randint(1, 20),
                entry_price=np.random.uniform(1, 10),
                current_price=np.random.uniform(1, 10),
                underlying_price=100 + np.random.uniform(-10, 10),
                implied_volatility=np.random.uniform(0.15, 0.45),
                greeks=OptionsGreeks(
                    delta=np.random.uniform(-1, 1),
                    gamma=np.random.uniform(0, 0.1),
                    vega=np.random.uniform(0, 1),
                    theta=np.random.uniform(-2, 0),
                    rho=np.random.uniform(-0.5, 0.5)
                )
            )
            risk_manager.positions.append(position)

        # Should complete in reasonable time
        import time
        start = time.time()
        greeks = risk_manager.calculate_portfolio_greeks()
        duration = time.time() - start

        assert duration < 1.0  # Should complete in under 1 second
        assert greeks["delta"] is not None

    def test_extreme_market_conditions(self):
        """Test strategies under extreme market conditions."""
        selector = OptionsStrategySelector(portfolio_value=1000000)

        # Extreme volatility
        extreme_signal = TradingSignal(
            source="market_crash",
            symbol="SPY",
            signal_type="bearish",
            strength=95,
            timeframe=1
        )

        extreme_market = {
            "symbol": "SPY",
            "price": 350,  # Down 20%
            "iv": 0.80,  # Extreme IV
            "vix": 65,  # Panic levels
            "drawdown": 25
        }

        result = selector.select_strategy(extreme_signal, extreme_market)

        assert result["name"] != "none"
        assert "defensive" in result.get("type", "") or "volatility" in result.get("type", "")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])