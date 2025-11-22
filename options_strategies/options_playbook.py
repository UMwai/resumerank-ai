"""
Options Strategy Playbook - Institutional Grade Options Trading Framework

This module provides comprehensive options strategies with specific entry/exit rules,
position sizing formulas, and risk parameters for institutional-grade trading.

Strategies Implemented:
1. Protective Puts - Portfolio insurance strategy
2. Covered Calls - Income generation strategy
3. Volatility Arbitrage - IV vs RV trading
4. Earnings Strategies - Event-driven options plays

Author: Quantitative Trading Team
Version: 1.0.0
"""

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification for adaptive strategies."""
    BULL = "bull"
    BEAR = "bear"
    NEUTRAL = "neutral"
    HIGH_VOL = "high_volatility"
    LOW_VOL = "low_volatility"


class OptionType(Enum):
    """Option type classification."""
    CALL = "call"
    PUT = "put"


@dataclass
class OptionsGreeks:
    """Option Greeks for risk management."""
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

    @property
    def dollar_delta(self) -> float:
        """Calculate dollar delta for position sizing."""
        return self.delta * 100  # Per contract (100 shares)


@dataclass
class OptionsPosition:
    """Represents an options position with all relevant parameters."""
    symbol: str
    option_type: OptionType
    strike: float
    expiration: datetime
    contracts: int
    entry_price: float
    current_price: float
    underlying_price: float
    implied_volatility: float
    greeks: OptionsGreeks
    entry_time: datetime = field(default_factory=datetime.now)

    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiration."""
        return (self.expiration - datetime.now()).days

    @property
    def pnl(self) -> float:
        """Calculate current P&L."""
        return (self.current_price - self.entry_price) * 100 * self.contracts

    @property
    def pnl_percent(self) -> float:
        """Calculate P&L percentage."""
        if self.entry_price > 0:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        return 0.0


class BlackScholesModel:
    """Enhanced Black-Scholes model for options pricing and Greeks calculation."""

    @staticmethod
    def calculate_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Calculate d1 and d2 for Black-Scholes formula."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @staticmethod
    def price_option(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: OptionType) -> float:
        """
        Calculate option price using Black-Scholes model.

        Args:
            S: Current price of underlying
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: Call or Put
        """
        from scipy.stats import norm

        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(S - K, 0)
            else:
                return max(K - S, 0)

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                        option_type: OptionType) -> OptionsGreeks:
        """Calculate all option Greeks."""
        from scipy.stats import norm

        if T <= 0:
            # At expiration, most Greeks are 0
            delta = 1.0 if (option_type == OptionType.CALL and S > K) else 0.0
            if option_type == OptionType.PUT and S < K:
                delta = -1.0
            return OptionsGreeks(delta=delta, gamma=0, vega=0, theta=0, rho=0)

        d1, d2 = BlackScholesModel.calculate_d1_d2(S, K, T, r, sigma)

        # Delta
        if option_type == OptionType.CALL:
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1

        # Gamma (same for calls and puts)
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        # Vega (same for calls and puts)
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divided by 100 for 1% change

        # Theta
        term1 = -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
            theta = (term1 + term2) / 365  # Daily theta
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)
            theta = (term1 + term2) / 365

        # Rho
        if option_type == OptionType.CALL:
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        return OptionsGreeks(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


class ProtectivePutStrategy:
    """
    Protective Put Strategy - Portfolio Insurance

    Systematic approach to deploying protective puts for downside protection
    while maintaining upside participation.
    """

    def __init__(self, portfolio_value: float, max_insurance_cost: float = 0.02):
        """
        Initialize Protective Put Strategy.

        Args:
            portfolio_value: Total portfolio value to protect
            max_insurance_cost: Maximum % of portfolio for insurance (default 2%)
        """
        self.portfolio_value = portfolio_value
        self.max_insurance_cost = max_insurance_cost
        logger.info(f"Initialized Protective Put Strategy for ${portfolio_value:,.0f} portfolio")

    def determine_deployment_conditions(self, vix: float, market_drawdown: float,
                                       earnings_days: int, fed_days: int) -> Dict[str, any]:
        """
        Determine when to deploy protective puts based on market conditions.

        Args:
            vix: Current VIX level
            market_drawdown: Current drawdown from recent high (%)
            earnings_days: Days until next major earnings season
            fed_days: Days until next Fed meeting

        Returns:
            Dictionary with deployment decision and parameters
        """
        deploy = False
        urgency = "low"
        sizing_multiplier = 1.0

        # High VIX threshold (>25 = deploy, >30 = urgent, >40 = max protection)
        if vix > 40:
            deploy = True
            urgency = "critical"
            sizing_multiplier = 2.0
        elif vix > 30:
            deploy = True
            urgency = "high"
            sizing_multiplier = 1.5
        elif vix > 25:
            deploy = True
            urgency = "moderate"
            sizing_multiplier = 1.2

        # Market drawdown triggers
        if market_drawdown > 10:
            deploy = True
            urgency = "high" if urgency == "low" else urgency
            sizing_multiplier = max(sizing_multiplier, 1.3)
        elif market_drawdown > 5:
            deploy = True
            urgency = "moderate" if urgency == "low" else urgency

        # Event protection
        if earnings_days < 14 or fed_days < 7:
            deploy = True
            urgency = "moderate" if urgency == "low" else urgency

        return {
            "deploy": deploy,
            "urgency": urgency,
            "sizing_multiplier": sizing_multiplier,
            "reasoning": self._generate_reasoning(vix, market_drawdown, earnings_days, fed_days)
        }

    def select_strike(self, current_price: float, volatility: float,
                     protection_level: str = "standard") -> Dict[str, float]:
        """
        Select optimal strike price for protective puts.

        Args:
            current_price: Current underlying price
            volatility: Implied volatility (annualized)
            protection_level: "aggressive" (ATM), "standard" (5% OTM), "conservative" (10% OTM)

        Returns:
            Dictionary with strike selection details
        """
        strike_percentages = {
            "aggressive": 0.00,   # At-the-money
            "standard": -0.05,    # 5% out-of-the-money
            "conservative": -0.10 # 10% out-of-the-money
        }

        strike_offset = strike_percentages.get(protection_level, -0.05)
        selected_strike = current_price * (1 + strike_offset)

        # Adjust for volatility - higher vol = closer to ATM
        if volatility > 0.30:  # High volatility
            selected_strike = current_price * (1 + strike_offset * 0.5)

        # Round to standard strike intervals
        if current_price < 50:
            selected_strike = round(selected_strike * 2) / 2  # $0.50 intervals
        elif current_price < 100:
            selected_strike = round(selected_strike)  # $1 intervals
        elif current_price < 500:
            selected_strike = round(selected_strike / 5) * 5  # $5 intervals
        else:
            selected_strike = round(selected_strike / 10) * 10  # $10 intervals

        return {
            "strike": selected_strike,
            "moneyness": (selected_strike / current_price - 1) * 100,
            "protection_starts_at": selected_strike,
            "max_loss_percent": abs(strike_offset) * 100
        }

    def select_expiration(self, volatility_regime: str, event_calendar: Dict) -> Dict[str, any]:
        """
        Select optimal expiration for protective puts.

        Args:
            volatility_regime: "low", "normal", "high", "extreme"
            event_calendar: Dict with upcoming events and dates

        Returns:
            Dictionary with expiration selection details
        """
        base_dte = {
            "low": 60,      # 60 days - longer dated in low vol
            "normal": 45,   # 45 days - standard protection
            "high": 30,     # 30 days - shorter in high vol
            "extreme": 21   # 21 days - very short term
        }

        selected_dte = base_dte.get(volatility_regime, 45)

        # Extend to cover known events
        for event, days_until in event_calendar.items():
            if days_until > selected_dte and days_until < 90:
                selected_dte = days_until + 7  # Cover event + 1 week

        # Find actual expiration date (3rd Friday typically)
        target_date = datetime.now() + timedelta(days=selected_dte)

        return {
            "target_dte": selected_dte,
            "expiration_date": target_date,
            "covers_events": [e for e, d in event_calendar.items() if d <= selected_dte],
            "theta_decay_acceleration": selected_dte < 30
        }

    def calculate_position_size(self, portfolio_value: float, put_price: float,
                               volatility: float, correlation: float = 0.85) -> Dict[str, any]:
        """
        Calculate optimal position size for protective puts.

        Args:
            portfolio_value: Value to protect
            put_price: Price per put contract
            volatility: Current implied volatility
            correlation: Correlation between puts and portfolio

        Returns:
            Dictionary with position sizing details
        """
        # Base insurance budget
        insurance_budget = portfolio_value * self.max_insurance_cost

        # Adjust for volatility - spend more in high vol
        if volatility > 0.30:
            insurance_budget *= 1.5
        elif volatility > 0.40:
            insurance_budget *= 2.0

        # Calculate number of contracts
        contract_value = put_price * 100
        num_contracts = int(insurance_budget / contract_value)

        # Calculate coverage ratio
        notional_protected = num_contracts * 100 * put_price / portfolio_value
        effective_protection = notional_protected * correlation

        return {
            "num_contracts": num_contracts,
            "total_cost": num_contracts * contract_value,
            "cost_as_percent": (num_contracts * contract_value / portfolio_value) * 100,
            "notional_protected": notional_protected * 100,
            "effective_protection": effective_protection * 100,
            "uncovered_exposure": max(0, 100 - effective_protection * 100)
        }

    def generate_rolling_strategy(self, current_dte: int, pnl_percent: float,
                                 underlying_move: float) -> Dict[str, any]:
        """
        Generate rolling strategy for existing protective puts.

        Args:
            current_dte: Days to expiration of current puts
            pnl_percent: Current P&L % on puts
            underlying_move: % move in underlying since put purchase

        Returns:
            Dictionary with rolling recommendations
        """
        action = "hold"
        reasoning = []

        # Roll if approaching expiration
        if current_dte <= 14:
            action = "roll"
            reasoning.append("Approaching expiration (< 14 DTE)")

        # Roll if puts are deep ITM and captured gains
        if pnl_percent > 100 and underlying_move < -5:
            action = "roll_up_and_out"
            reasoning.append("Captured significant gains, roll to higher strike")

        # Roll if puts are far OTM and cheap to adjust
        if pnl_percent < -70 and underlying_move > 5:
            action = "roll_down"
            reasoning.append("Puts too far OTM, roll to lower strike")

        # Let expire if near worthless
        if current_dte <= 7 and pnl_percent < -90:
            action = "let_expire"
            reasoning.append("Near worthless, let expire")

        return {
            "action": action,
            "reasoning": " | ".join(reasoning),
            "new_dte_target": 45 if "roll" in action else None,
            "strike_adjustment": self._determine_strike_adjustment(action, underlying_move)
        }

    def _generate_reasoning(self, vix: float, drawdown: float,
                           earnings_days: int, fed_days: int) -> str:
        """Generate reasoning for deployment decision."""
        reasons = []
        if vix > 25:
            reasons.append(f"VIX elevated at {vix:.1f}")
        if drawdown > 5:
            reasons.append(f"Market drawdown of {drawdown:.1f}%")
        if earnings_days < 14:
            reasons.append(f"Earnings season in {earnings_days} days")
        if fed_days < 7:
            reasons.append(f"Fed meeting in {fed_days} days")
        return " | ".join(reasons) if reasons else "No immediate triggers"

    def _determine_strike_adjustment(self, action: str, underlying_move: float) -> float:
        """Determine strike adjustment for rolling."""
        if "up" in action:
            return min(underlying_move * 0.5, 5.0)  # Roll up half the move, max 5%
        elif "down" in action:
            return max(underlying_move * 0.5, -5.0)  # Roll down half the move, max -5%
        return 0.0


class CoveredCallStrategy:
    """
    Covered Call Strategy - Income Generation

    Systematic approach to writing covered calls for premium income
    while managing assignment risk.
    """

    def __init__(self, position_size: int, target_monthly_income: float = 0.02):
        """
        Initialize Covered Call Strategy.

        Args:
            position_size: Number of shares owned (must be multiple of 100)
            target_monthly_income: Target income as % of position (default 2%)
        """
        self.position_size = position_size
        self.contracts_available = position_size // 100
        self.target_monthly_income = target_monthly_income
        logger.info(f"Initialized Covered Call Strategy for {position_size} shares")

    def select_strike(self, current_price: float, volatility: float,
                     technical_resistance: float, iv_rank: float) -> Dict[str, any]:
        """
        Select optimal strike price for covered calls.

        Args:
            current_price: Current stock price
            volatility: 30-day implied volatility
            technical_resistance: Key technical resistance level
            iv_rank: IV rank (0-100)

        Returns:
            Dictionary with strike selection details
        """
        # Base strike selection - start 2-5% OTM
        if iv_rank > 70:  # High IV rank - more aggressive
            base_otm = 0.02  # 2% OTM
        elif iv_rank > 30:  # Normal IV
            base_otm = 0.03  # 3% OTM
        else:  # Low IV
            base_otm = 0.05  # 5% OTM

        # Adjust for volatility
        volatility_adjustment = volatility * 0.5  # Half of volatility
        target_strike = current_price * (1 + base_otm + volatility_adjustment)

        # Respect technical levels
        if technical_resistance > current_price:
            target_strike = min(target_strike, technical_resistance * 1.01)

        # Round to standard strikes
        if current_price < 50:
            target_strike = round(target_strike * 2) / 2
        elif current_price < 100:
            target_strike = round(target_strike)
        else:
            target_strike = round(target_strike / 5) * 5

        # Calculate expected return
        premium_estimate = self._estimate_premium(current_price, target_strike, volatility)

        return {
            "strike": target_strike,
            "moneyness": (target_strike / current_price - 1) * 100,
            "premium_estimate": premium_estimate,
            "annualized_return": (premium_estimate / current_price) * 12 * 100,
            "assignment_probability": self._calculate_assignment_probability(
                current_price, target_strike, volatility, 30
            )
        }

    def select_expiration_cycle(self, volatility: float, earnings_date: Optional[datetime],
                              dividend_date: Optional[datetime]) -> Dict[str, any]:
        """
        Select optimal expiration cycle for covered calls.

        Args:
            volatility: Current implied volatility
            earnings_date: Next earnings date
            dividend_date: Next ex-dividend date

        Returns:
            Dictionary with expiration cycle selection
        """
        # Default cycles
        cycles = {
            "weekly": 7,
            "bi_weekly": 14,
            "monthly": 30,
            "45_days": 45,
            "quarterly": 90
        }

        # Select based on volatility
        if volatility > 0.40:  # High volatility
            selected_cycle = "weekly"
            reasoning = "High volatility favors short-term premium capture"
        elif volatility > 0.25:  # Moderate volatility
            selected_cycle = "monthly"
            reasoning = "Moderate volatility suits monthly cycle"
        else:  # Low volatility
            selected_cycle = "45_days"
            reasoning = "Low volatility requires longer dated for adequate premium"

        selected_dte = cycles[selected_cycle]

        # Avoid earnings if within cycle
        if earnings_date:
            days_to_earnings = (earnings_date - datetime.now()).days
            if 0 < days_to_earnings < selected_dte:
                selected_dte = days_to_earnings - 2
                reasoning += f" | Adjusted to expire before earnings"

        # Consider dividend capture
        if dividend_date:
            days_to_dividend = (dividend_date - datetime.now()).days
            if 0 < days_to_dividend < selected_dte + 7:
                reasoning += f" | Dividend in {days_to_dividend} days"

        return {
            "cycle": selected_cycle,
            "target_dte": selected_dte,
            "expiration_date": datetime.now() + timedelta(days=selected_dte),
            "reasoning": reasoning,
            "expected_cycles_per_year": 365 / selected_dte
        }

    def determine_holdings_to_write(self, holdings: List[Dict],
                                   iv_threshold: float = 50) -> List[Dict]:
        """
        Determine which holdings to write calls against.

        Args:
            holdings: List of holdings with metrics
            iv_threshold: Minimum IV rank to consider

        Returns:
            List of holdings suitable for covered calls
        """
        suitable_holdings = []

        for holding in holdings:
            score = 0
            reasons = []

            # Check IV rank
            if holding["iv_rank"] > iv_threshold:
                score += 30
                reasons.append(f"High IV rank ({holding['iv_rank']})")

            # Check volatility
            if 0.20 < holding["volatility"] < 0.50:
                score += 20
                reasons.append("Optimal volatility range")

            # Check trend
            if holding["trend"] in ["neutral", "mild_bullish"]:
                score += 25
                reasons.append(f"Suitable trend ({holding['trend']})")

            # Check liquidity
            if holding["option_volume"] > 1000:
                score += 15
                reasons.append("Good option liquidity")

            # Check dividend yield
            if holding.get("dividend_yield", 0) > 0.02:
                score += 10
                reasons.append(f"Dividend yield {holding['dividend_yield']:.1%}")

            if score >= 50:  # Threshold for suitability
                suitable_holdings.append({
                    "symbol": holding["symbol"],
                    "score": score,
                    "reasons": reasons,
                    "recommended_allocation": self._calculate_allocation(score)
                })

        return sorted(suitable_holdings, key=lambda x: x["score"], reverse=True)

    def manage_assignment_risk(self, dte: int, moneyness: float,
                              delta: float, pnl: float) -> Dict[str, any]:
        """
        Manage assignment risk for existing covered calls.

        Args:
            dte: Days to expiration
            moneyness: Current moneyness (%)
            delta: Current delta of short call
            pnl: Current P&L on short call

        Returns:
            Dictionary with risk management actions
        """
        action = "hold"
        reasoning = []

        # Check if call is ITM
        if moneyness < 0:  # ITM
            if dte > 14:
                action = "roll_up_and_out"
                reasoning.append("ITM with time to roll")
            elif dte > 7:
                action = "evaluate_rolling"
                reasoning.append("ITM approaching expiration")
            else:
                if pnl < -200:  # Significant loss on call
                    action = "roll_or_accept_assignment"
                    reasoning.append("Evaluate if assignment acceptable")
                else:
                    action = "let_assign"
                    reasoning.append("Accept assignment for profit")

        # High delta risk
        if delta < -0.70 and dte > 7:
            action = "consider_rolling"
            reasoning.append(f"High delta risk ({delta:.2f})")

        # Profitable close opportunity
        if pnl > 0.75 * abs(pnl) and dte > 14:  # Captured 75% of max profit
            action = "close_and_reset"
            reasoning.append("Captured majority of profit")

        return {
            "action": action,
            "reasoning": " | ".join(reasoning),
            "urgency": "high" if moneyness < -2 and dte < 7 else "normal",
            "estimated_assignment_prob": self._estimate_assignment_probability_current(
                moneyness, dte, delta
            )
        }

    def calculate_target_premium(self, stock_price: float,
                                target_return: float = 0.02) -> float:
        """Calculate target premium for desired return."""
        return stock_price * target_return

    def _estimate_premium(self, stock_price: float, strike: float,
                         volatility: float, dte: int = 30) -> float:
        """Estimate call premium using simplified model."""
        moneyness = (strike - stock_price) / stock_price
        time_value = volatility * np.sqrt(dte / 365) * stock_price * 0.4
        intrinsic = max(0, stock_price - strike)

        # Adjust time value for moneyness
        if moneyness > 0:  # OTM
            time_value *= np.exp(-2 * moneyness / volatility)

        return intrinsic + time_value

    def _calculate_assignment_probability(self, stock_price: float, strike: float,
                                         volatility: float, dte: int) -> float:
        """Calculate probability of assignment at expiration."""
        from scipy.stats import norm

        if dte <= 0:
            return 1.0 if stock_price >= strike else 0.0

        # Use Black-Scholes N(d2) as proxy for assignment probability
        r = 0.05  # Risk-free rate
        T = dte / 365
        d1 = (np.log(stock_price / strike) + (r + 0.5 * volatility ** 2) * T) / (volatility * np.sqrt(T))
        d2 = d1 - volatility * np.sqrt(T)

        return norm.cdf(d2)

    def _calculate_allocation(self, score: float) -> float:
        """Calculate recommended allocation based on score."""
        if score >= 80:
            return 1.0  # Full allocation
        elif score >= 65:
            return 0.75
        elif score >= 50:
            return 0.50
        else:
            return 0.25

    def _estimate_assignment_probability_current(self, moneyness: float,
                                                dte: int, delta: float) -> float:
        """Estimate current assignment probability."""
        if moneyness < 0:  # ITM
            base_prob = 0.70
            if dte < 7:
                base_prob = 0.90
            elif dte < 14:
                base_prob = 0.80
        else:  # OTM
            base_prob = abs(delta)  # Delta approximation

        return min(1.0, max(0.0, base_prob))


class VolatilityArbitrageStrategy:
    """
    Volatility Arbitrage Strategy

    Trade the difference between implied and realized volatility,
    including VIX futures and calendar spreads.
    """

    def __init__(self, capital_allocated: float, max_vega_exposure: float = 10000):
        """
        Initialize Volatility Arbitrage Strategy.

        Args:
            capital_allocated: Capital allocated to vol strategies
            max_vega_exposure: Maximum vega exposure allowed
        """
        self.capital_allocated = capital_allocated
        self.max_vega_exposure = max_vega_exposure
        logger.info(f"Initialized Volatility Arbitrage Strategy with ${capital_allocated:,.0f}")

    def identify_iv_rv_divergence(self, symbol: str, iv_30d: float,
                                 rv_30d: float, rv_60d: float) -> Dict[str, any]:
        """
        Identify tradeable divergence between implied and realized volatility.

        Args:
            symbol: Underlying symbol
            iv_30d: 30-day implied volatility
            rv_30d: 30-day realized volatility
            rv_60d: 60-day realized volatility

        Returns:
            Dictionary with trade signals
        """
        # Calculate divergence
        iv_rv_spread = iv_30d - rv_30d
        iv_rv_ratio = iv_30d / rv_30d if rv_30d > 0 else 0

        # Historical context
        rv_trend = (rv_30d - rv_60d) / rv_60d if rv_60d > 0 else 0

        # Generate signals
        signal = "neutral"
        confidence = 0
        strategy = None

        # Significant IV premium (IV > RV by 25% or 5 vol points)
        if iv_rv_spread > 0.05 or iv_rv_ratio > 1.25:
            signal = "sell_volatility"
            confidence = min(100, 50 + (iv_rv_spread * 1000))
            strategy = "short_straddle" if confidence > 70 else "iron_condor"

        # Significant IV discount (RV > IV by 20% or 3 vol points)
        elif iv_rv_spread < -0.03 or iv_rv_ratio < 0.80:
            signal = "buy_volatility"
            confidence = min(100, 50 + abs(iv_rv_spread * 1000))
            strategy = "long_straddle" if confidence > 70 else "long_strangle"

        # Check if RV is trending
        if abs(rv_trend) > 0.20:
            confidence *= 0.8  # Reduce confidence in trending environment

        return {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "strategy": strategy,
            "metrics": {
                "iv_30d": iv_30d,
                "rv_30d": rv_30d,
                "iv_rv_spread": iv_rv_spread,
                "iv_rv_ratio": iv_rv_ratio,
                "rv_trend": rv_trend
            },
            "entry_threshold": self._calculate_entry_threshold(iv_rv_spread),
            "exit_threshold": self._calculate_exit_threshold(iv_rv_spread)
        }

    def analyze_vix_futures(self, spot_vix: float, front_month: float,
                          second_month: float, days_to_expiry: int) -> Dict[str, any]:
        """
        Analyze VIX futures for contango/backwardation trades.

        Args:
            spot_vix: Spot VIX level
            front_month: Front month VIX future price
            second_month: Second month VIX future price
            days_to_expiry: Days until front month expiry

        Returns:
            Dictionary with VIX futures analysis
        """
        # Calculate term structure
        front_premium = (front_month - spot_vix) / spot_vix
        roll_yield = (second_month - front_month) / front_month
        daily_roll = roll_yield / 30  # Approximate daily roll

        # Determine market structure
        if front_month > spot_vix and second_month > front_month:
            structure = "contango"
            if roll_yield > 0.05:  # > 5% roll cost
                signal = "short_vix_futures"
                confidence = min(100, 50 + roll_yield * 1000)
            else:
                signal = "mild_contango"
                confidence = 30

        elif front_month < spot_vix and second_month < front_month:
            structure = "backwardation"
            if abs(roll_yield) > 0.05:
                signal = "long_vix_futures"
                confidence = min(100, 50 + abs(roll_yield) * 1000)
            else:
                signal = "mild_backwardation"
                confidence = 30
        else:
            structure = "mixed"
            signal = "neutral"
            confidence = 0

        # Position sizing based on roll yield
        position_size = self._calculate_vix_position_size(
            roll_yield, spot_vix, self.capital_allocated
        )

        return {
            "structure": structure,
            "signal": signal,
            "confidence": confidence,
            "metrics": {
                "spot_vix": spot_vix,
                "front_month": front_month,
                "second_month": second_month,
                "front_premium": front_premium * 100,
                "roll_yield": roll_yield * 100,
                "daily_roll": daily_roll * 100,
                "days_to_expiry": days_to_expiry
            },
            "position_size": position_size,
            "risk_metrics": {
                "max_loss": position_size * 0.50,  # VIX can spike 50%
                "expected_return": position_size * daily_roll * days_to_expiry,
                "risk_reward": abs((daily_roll * days_to_expiry) / 0.50)
            }
        }

    def design_calendar_spread(self, symbol: str, iv_term_structure: Dict[int, float],
                              current_price: float) -> Dict[str, any]:
        """
        Design calendar spread for volatility arbitrage.

        Args:
            symbol: Underlying symbol
            iv_term_structure: Dict mapping DTE to IV {30: 0.25, 60: 0.28, 90: 0.30}
            current_price: Current underlying price

        Returns:
            Dictionary with calendar spread design
        """
        # Find optimal calendar spread opportunity
        best_spread = None
        best_edge = 0

        dates = sorted(iv_term_structure.keys())
        for i in range(len(dates) - 1):
            near_dte = dates[i]
            far_dte = dates[i + 1]

            near_iv = iv_term_structure[near_dte]
            far_iv = iv_term_structure[far_dte]

            # Calculate expected edge
            iv_diff = far_iv - near_iv
            time_ratio = np.sqrt(far_dte / near_dte)
            expected_iv_ratio = time_ratio * 1.1  # Expect 10% premium for time

            edge = (far_iv / near_iv) / expected_iv_ratio - 1

            if abs(edge) > abs(best_edge):
                best_edge = edge
                best_spread = {
                    "near_dte": near_dte,
                    "far_dte": far_dte,
                    "near_iv": near_iv,
                    "far_iv": far_iv,
                    "iv_diff": iv_diff,
                    "edge": edge
                }

        if best_spread and abs(best_edge) > 0.05:  # 5% edge threshold
            # Design the trade
            if best_edge > 0:  # Far month overpriced
                trade = "sell_calendar"
                near_leg = "buy"
                far_leg = "sell"
            else:  # Near month overpriced
                trade = "buy_calendar"
                near_leg = "sell"
                far_leg = "buy"

            # Calculate strikes (ATM for calendar spreads)
            strike = self._round_to_strike(current_price, current_price)

            # Position sizing
            max_loss = self._estimate_calendar_max_loss(
                current_price, strike, best_spread["near_iv"], best_spread["far_iv"]
            )
            num_spreads = int(self.capital_allocated * 0.05 / max_loss)  # Risk 5% of capital

            return {
                "signal": trade,
                "structure": {
                    "near_leg": {
                        "action": near_leg,
                        "dte": best_spread["near_dte"],
                        "strike": strike,
                        "iv": best_spread["near_iv"]
                    },
                    "far_leg": {
                        "action": far_leg,
                        "dte": best_spread["far_dte"],
                        "strike": strike,
                        "iv": best_spread["far_iv"]
                    }
                },
                "edge": best_edge * 100,
                "num_spreads": num_spreads,
                "max_loss": max_loss * num_spreads,
                "breakeven_points": self._calculate_calendar_breakevens(
                    current_price, strike, best_spread["near_iv"]
                ),
                "management": {
                    "profit_target": max_loss * 0.25,  # 25% of max loss
                    "stop_loss": max_loss * 0.50,  # 50% of max loss
                    "adjustment_points": [current_price * 0.97, current_price * 1.03]
                }
            }

        return {"signal": "no_opportunity", "reason": "Insufficient edge in term structure"}

    def calculate_position_limits(self, portfolio_vega: float,
                                 portfolio_gamma: float) -> Dict[str, float]:
        """
        Calculate position limits for volatility strategies.

        Args:
            portfolio_vega: Current portfolio vega
            portfolio_gamma: Current portfolio gamma

        Returns:
            Dictionary with position limits
        """
        # Vega limits
        vega_remaining = self.max_vega_exposure - abs(portfolio_vega)
        vega_limit_contracts = vega_remaining / 50  # Assume 50 vega per contract

        # Gamma limits (gamma-neutral bands)
        gamma_limit = self.capital_allocated * 0.0001  # 0.01% of capital
        gamma_remaining = gamma_limit - abs(portfolio_gamma)
        gamma_limit_contracts = gamma_remaining / 0.05  # Assume 0.05 gamma per contract

        # Capital limits
        max_premium_spend = self.capital_allocated * 0.10  # 10% on premium
        max_margin_requirement = self.capital_allocated * 0.30  # 30% margin max

        return {
            "vega_limit": vega_remaining,
            "vega_contracts": max(0, int(vega_limit_contracts)),
            "gamma_limit": gamma_remaining,
            "gamma_contracts": max(0, int(gamma_limit_contracts)),
            "max_premium": max_premium_spend,
            "max_margin": max_margin_requirement,
            "suggested_trades": min(
                int(vega_limit_contracts),
                int(gamma_limit_contracts),
                int(max_premium_spend / 500)  # Assume $500 per spread
            )
        }

    def _calculate_entry_threshold(self, iv_rv_spread: float) -> float:
        """Calculate entry threshold for IV/RV trade."""
        return abs(iv_rv_spread) * 0.5  # Enter when spread narrows by 50%

    def _calculate_exit_threshold(self, iv_rv_spread: float) -> float:
        """Calculate exit threshold for IV/RV trade."""
        return abs(iv_rv_spread) * 0.2  # Exit when spread narrows by 80%

    def _calculate_vix_position_size(self, roll_yield: float, vix_level: float,
                                    capital: float) -> float:
        """Calculate VIX futures position size."""
        # Base size on roll yield and VIX level
        if vix_level > 30:  # High VIX - reduce size
            size_multiplier = 0.5
        elif vix_level > 20:
            size_multiplier = 0.75
        else:
            size_multiplier = 1.0

        # Size based on expected roll capture
        base_size = capital * min(0.10, abs(roll_yield))  # Max 10% of capital
        return base_size * size_multiplier

    def _round_to_strike(self, price: float, target: float) -> float:
        """Round to nearest standard strike."""
        if price < 50:
            return round(target * 2) / 2
        elif price < 100:
            return round(target)
        elif price < 500:
            return round(target / 5) * 5
        else:
            return round(target / 10) * 10

    def _estimate_calendar_max_loss(self, stock_price: float, strike: float,
                                   near_iv: float, far_iv: float) -> float:
        """Estimate max loss for calendar spread."""
        # Simplified estimation
        near_premium = stock_price * near_iv * np.sqrt(30/365) * 0.4
        far_premium = stock_price * far_iv * np.sqrt(60/365) * 0.4
        return abs(far_premium - near_premium) * 100  # Per spread

    def _calculate_calendar_breakevens(self, stock_price: float,
                                      strike: float, iv: float) -> List[float]:
        """Calculate breakeven points for calendar spread."""
        # Approximate breakevens at 1 standard deviation
        std_dev = stock_price * iv * np.sqrt(30/365)
        return [strike - std_dev, strike + std_dev]


class EarningsStrategy:
    """
    Earnings Strategy - Event-Driven Options Plays

    Systematic approach to trading earnings announcements using
    straddles, strangles, and directional plays.
    """

    def __init__(self, capital_per_trade: float = 5000):
        """
        Initialize Earnings Strategy.

        Args:
            capital_per_trade: Capital allocated per earnings trade
        """
        self.capital_per_trade = capital_per_trade
        logger.info(f"Initialized Earnings Strategy with ${capital_per_trade:,.0f} per trade")

    def analyze_earnings_setup(self, symbol: str, days_to_earnings: int,
                              iv_current: float, iv_historical: List[float],
                              historical_moves: List[float]) -> Dict[str, any]:
        """
        Analyze earnings setup and recommend strategy.

        Args:
            symbol: Stock symbol
            days_to_earnings: Days until earnings
            iv_current: Current implied volatility
            iv_historical: Historical IV for past 4 earnings
            historical_moves: Historical earnings moves (past 4)

        Returns:
            Dictionary with earnings analysis and strategy
        """
        # Calculate IV rank and percentile
        iv_rank = self._calculate_iv_rank(iv_current, iv_historical)

        # Calculate expected move
        expected_move = iv_current * np.sqrt(days_to_earnings / 365) * np.sqrt(8 / np.pi)

        # Calculate historical move statistics
        avg_move = np.mean(np.abs(historical_moves))
        max_move = np.max(np.abs(historical_moves))
        move_consistency = np.std(historical_moves) / avg_move if avg_move > 0 else 0

        # Determine strategy
        if iv_rank > 80:  # Very high IV rank
            if expected_move > avg_move * 1.3:
                strategy = "short_strangle"
                confidence = 75
                reasoning = "IV significantly elevated vs historical moves"
            else:
                strategy = "iron_condor"
                confidence = 60
                reasoning = "High IV but move expectations reasonable"

        elif iv_rank > 50:  # Moderate IV rank
            if move_consistency < 0.3:  # Consistent mover
                strategy = "atm_straddle"
                confidence = 65
                reasoning = "Consistent historical moves with moderate IV"
            else:
                strategy = "strangle"
                confidence = 55
                reasoning = "Variable moves suggest wider strikes"

        else:  # Low IV rank
            if avg_move > expected_move * 1.2:
                strategy = "long_straddle"
                confidence = 70
                reasoning = "Historical moves exceed current expectations"
            else:
                strategy = "avoid"
                confidence = 0
                reasoning = "Insufficient edge - IV too low"

        return {
            "symbol": symbol,
            "strategy": strategy,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "days_to_earnings": days_to_earnings,
                "iv_current": iv_current * 100,
                "iv_rank": iv_rank,
                "expected_move_pct": expected_move * 100,
                "avg_historical_move": avg_move * 100,
                "max_historical_move": max_move * 100,
                "move_consistency": move_consistency
            },
            "trade_details": self._generate_trade_details(
                strategy, expected_move, self.capital_per_trade
            )
        }

    def design_straddle_strangle(self, stock_price: float, iv: float,
                                expected_move: float, capital: float,
                                use_straddle: bool = True) -> Dict[str, any]:
        """
        Design ATM straddle or strangle position.

        Args:
            stock_price: Current stock price
            iv: Implied volatility
            expected_move: Expected move percentage
            capital: Capital to allocate
            use_straddle: True for straddle, False for strangle

        Returns:
            Dictionary with position details
        """
        if use_straddle:
            # ATM straddle
            strike = self._round_to_strike(stock_price, stock_price)
            put_strike = strike
            call_strike = strike
            position_type = "straddle"
        else:
            # Strangle with strikes at expected move
            put_strike = self._round_to_strike(
                stock_price,
                stock_price * (1 - expected_move)
            )
            call_strike = self._round_to_strike(
                stock_price,
                stock_price * (1 + expected_move)
            )
            position_type = "strangle"

        # Estimate premiums
        put_premium = self._estimate_option_premium(
            stock_price, put_strike, iv, 3, OptionType.PUT
        )
        call_premium = self._estimate_option_premium(
            stock_price, call_strike, iv, 3, OptionType.CALL
        )
        total_premium = put_premium + call_premium

        # Calculate number of positions
        num_positions = int(capital / (total_premium * 100))

        # Calculate breakevens
        if use_straddle:
            upper_breakeven = strike + total_premium
            lower_breakeven = strike - total_premium
        else:
            upper_breakeven = call_strike + total_premium
            lower_breakeven = put_strike - total_premium

        # Required move to profit
        required_move = min(
            abs(upper_breakeven - stock_price),
            abs(lower_breakeven - stock_price)
        ) / stock_price

        return {
            "position_type": position_type,
            "legs": {
                "put": {
                    "strike": put_strike,
                    "premium": put_premium,
                    "contracts": num_positions
                },
                "call": {
                    "strike": call_strike,
                    "premium": call_premium,
                    "contracts": num_positions
                }
            },
            "cost": total_premium * 100 * num_positions,
            "breakevens": {
                "upper": upper_breakeven,
                "lower": lower_breakeven,
                "required_move_pct": required_move * 100
            },
            "risk_reward": {
                "max_loss": total_premium * 100 * num_positions,
                "profit_at_expected": self._calculate_straddle_profit(
                    stock_price * (1 + expected_move),
                    call_strike, put_strike, total_premium
                ) * num_positions,
                "profit_at_2x": self._calculate_straddle_profit(
                    stock_price * (1 + 2 * expected_move),
                    call_strike, put_strike, total_premium
                ) * num_positions
            }
        }

    def determine_entry_timing(self, days_to_earnings: int, iv_curve: Dict[int, float],
                             historical_iv_pattern: str) -> Dict[str, any]:
        """
        Determine optimal entry timing for earnings trade.

        Args:
            days_to_earnings: Days until earnings
            iv_curve: IV by days to earnings {5: 0.45, 3: 0.50, 1: 0.60}
            historical_iv_pattern: "early_ramp", "late_spike", "gradual"

        Returns:
            Dictionary with entry timing recommendation
        """
        recommendations = {
            "early_ramp": {
                "ideal_entry": max(5, days_to_earnings - 2),
                "reasoning": "IV ramps early - enter close to earnings",
                "iv_target": "Wait for IV > 75th percentile"
            },
            "late_spike": {
                "ideal_entry": min(7, days_to_earnings),
                "reasoning": "IV spikes late - enter 1 week out",
                "iv_target": "Enter before major IV expansion"
            },
            "gradual": {
                "ideal_entry": min(10, days_to_earnings),
                "reasoning": "Gradual IV build - enter 10 days out",
                "iv_target": "Capture steady IV expansion"
            }
        }

        rec = recommendations.get(historical_iv_pattern, recommendations["gradual"])

        # Calculate expected IV at entry
        if rec["ideal_entry"] in iv_curve:
            expected_iv = iv_curve[rec["ideal_entry"]]
        else:
            # Interpolate
            expected_iv = np.mean(list(iv_curve.values()))

        return {
            "ideal_entry_dte": rec["ideal_entry"],
            "entry_date": datetime.now() + timedelta(days=days_to_earnings - rec["ideal_entry"]),
            "reasoning": rec["reasoning"],
            "iv_target": rec["iv_target"],
            "expected_iv": expected_iv * 100,
            "urgency": "high" if rec["ideal_entry"] >= days_to_earnings - 1 else "normal"
        }

    def calculate_position_size(self, confidence: float, kelly_fraction: float,
                               max_loss: float) -> Dict[str, float]:
        """
        Calculate position size for earnings trade.

        Args:
            confidence: Strategy confidence (0-100)
            kelly_fraction: Kelly criterion fraction
            max_loss: Maximum loss of position

        Returns:
            Dictionary with position sizing details
        """
        # Base allocation on confidence
        confidence_multiplier = confidence / 100

        # Apply Kelly fraction with safety factor
        kelly_adjusted = kelly_fraction * 0.25  # Use 25% Kelly

        # Calculate position size
        base_size = self.capital_per_trade
        adjusted_size = base_size * confidence_multiplier * (1 + kelly_adjusted)

        # Apply maximum constraints
        max_size = min(adjusted_size, self.capital_per_trade * 2)  # Max 2x base
        final_size = min(max_size, max_loss * 5)  # Max loss should be 20% of position

        return {
            "recommended_size": final_size,
            "base_size": base_size,
            "confidence_adjustment": confidence_multiplier,
            "kelly_adjustment": kelly_adjusted,
            "size_as_pct_of_max": (final_size / (self.capital_per_trade * 2)) * 100,
            "number_of_contracts": int(final_size / max_loss * 100)
        }

    def generate_exit_plan(self, strategy: str, entry_price: float,
                         expected_move: float, days_to_earnings: int) -> Dict[str, any]:
        """
        Generate exit plan for earnings position.

        Args:
            strategy: Strategy type (straddle, strangle, etc.)
            entry_price: Entry price of position
            expected_move: Expected move percentage
            days_to_earnings: Days until earnings when entered

        Returns:
            Dictionary with exit plan
        """
        if strategy in ["long_straddle", "atm_straddle", "strangle"]:
            exit_plan = {
                "before_earnings": {
                    "condition": "If IV contracts or profit > 20%",
                    "timing": "1-2 hours before close on earnings day",
                    "target": entry_price * 1.20
                },
                "after_earnings": {
                    "condition": "After earnings release",
                    "timing": "First 30 minutes after open",
                    "rules": [
                        "Exit if move < 50% of expected",
                        "Hold if move > expected for momentum",
                        "Exit by noon regardless"
                    ]
                },
                "stop_loss": entry_price * 0.50,  # 50% stop loss
                "time_stop": days_to_earnings + 1  # Exit day after earnings
            }

        elif strategy in ["short_strangle", "iron_condor"]:
            exit_plan = {
                "before_earnings": {
                    "condition": "If profit > 50% of max or breached",
                    "timing": "Anytime before earnings",
                    "target": entry_price * 0.50
                },
                "after_earnings": {
                    "condition": "If within profit zone",
                    "timing": "Hold for theta decay 1-2 days",
                    "rules": [
                        "Exit if profit > 75% of max",
                        "Exit if challenged on either side",
                        "Let expire if > 80% profit"
                    ]
                },
                "stop_loss": entry_price * 2.0,  # 100% loss (200% of credit)
                "adjustment": "Roll untested side if breached"
            }
        else:
            exit_plan = {
                "general": "No position recommended",
                "reasoning": strategy
            }

        return exit_plan

    def _calculate_iv_rank(self, current_iv: float, historical_iv: List[float]) -> float:
        """Calculate IV rank (0-100)."""
        if not historical_iv:
            return 50

        min_iv = min(historical_iv)
        max_iv = max(historical_iv)

        if max_iv == min_iv:
            return 50

        return ((current_iv - min_iv) / (max_iv - min_iv)) * 100

    def _generate_trade_details(self, strategy: str, expected_move: float,
                               capital: float) -> Dict[str, any]:
        """Generate specific trade details based on strategy."""
        details = {
            "strategy": strategy,
            "capital_allocated": capital
        }

        if strategy == "short_strangle":
            details["strikes"] = {
                "put": f"{(1 - expected_move * 1.5) * 100:.1f}% OTM",
                "call": f"{(expected_move * 1.5) * 100:.1f}% OTM"
            }
            details["max_profit"] = capital * 0.15  # Estimate 15% credit

        elif strategy == "atm_straddle":
            details["strikes"] = "ATM for both legs"
            details["cost_estimate"] = capital
            details["profit_target"] = capital * 0.30

        elif strategy == "iron_condor":
            details["strikes"] = {
                "short_put": f"{(1 - expected_move) * 100:.1f}% OTM",
                "long_put": f"{(1 - expected_move * 1.5) * 100:.1f}% OTM",
                "short_call": f"{expected_move * 100:.1f}% OTM",
                "long_call": f"{(expected_move * 1.5) * 100:.1f}% OTM"
            }
            details["max_profit"] = capital * 0.20

        return details

    def _round_to_strike(self, price: float, target: float) -> float:
        """Round to nearest standard strike."""
        if price < 50:
            return round(target * 2) / 2
        elif price < 100:
            return round(target)
        elif price < 500:
            return round(target / 5) * 5
        else:
            return round(target / 10) * 10

    def _estimate_option_premium(self, stock_price: float, strike: float,
                                iv: float, dte: int, option_type: OptionType) -> float:
        """Estimate option premium using simplified Black-Scholes."""
        r = 0.05  # Risk-free rate
        T = dte / 365

        return BlackScholesModel.price_option(
            stock_price, strike, T, r, iv, option_type
        )

    def _calculate_straddle_profit(self, final_price: float, call_strike: float,
                                  put_strike: float, total_premium: float) -> float:
        """Calculate profit for straddle/strangle at given price."""
        call_value = max(0, final_price - call_strike)
        put_value = max(0, put_strike - final_price)
        return (call_value + put_value - total_premium) * 100


class OptionsRiskManager:
    """
    Comprehensive risk management for options strategies.
    Monitors positions, calculates portfolio Greeks, and enforces risk limits.
    """

    def __init__(self, portfolio_value: float, max_portfolio_theta: float = -1000):
        """
        Initialize Options Risk Manager.

        Args:
            portfolio_value: Total portfolio value
            max_portfolio_theta: Maximum daily theta decay allowed
        """
        self.portfolio_value = portfolio_value
        self.max_portfolio_theta = max_portfolio_theta
        self.positions: List[OptionsPosition] = []
        logger.info(f"Initialized Options Risk Manager for ${portfolio_value:,.0f}")

    def calculate_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate aggregate portfolio Greeks."""
        total_delta = sum(p.greeks.delta * p.contracts for p in self.positions)
        total_gamma = sum(p.greeks.gamma * p.contracts for p in self.positions)
        total_vega = sum(p.greeks.vega * p.contracts for p in self.positions)
        total_theta = sum(p.greeks.theta * p.contracts for p in self.positions)

        return {
            "delta": total_delta,
            "gamma": total_gamma,
            "vega": total_vega,
            "theta": total_theta,
            "delta_dollars": total_delta * 100,  # Dollar exposure
            "theta_daily": total_theta,
            "vega_1pct": total_vega  # Exposure to 1% IV change
        }

    def check_risk_limits(self, new_position: OptionsPosition) -> Dict[str, any]:
        """Check if new position would breach risk limits."""
        # Calculate new portfolio Greeks
        new_greeks = self.calculate_portfolio_greeks()
        new_greeks["delta"] += new_position.greeks.delta * new_position.contracts
        new_greeks["theta"] += new_position.greeks.theta * new_position.contracts
        new_greeks["vega"] += new_position.greeks.vega * new_position.contracts

        violations = []

        # Check theta limit
        if new_greeks["theta"] < self.max_portfolio_theta:
            violations.append(f"Theta limit breached: {new_greeks['theta']:.2f}")

        # Check delta exposure (max 20% of portfolio)
        if abs(new_greeks["delta_dollars"]) > self.portfolio_value * 0.20:
            violations.append(f"Delta exposure too high: ${new_greeks['delta_dollars']:.2f}")

        # Check vega exposure (max 5% portfolio move on 10% IV change)
        if abs(new_greeks["vega"] * 10) > self.portfolio_value * 0.05:
            violations.append(f"Vega exposure too high: ${new_greeks['vega'] * 10:.2f}")

        return {
            "approved": len(violations) == 0,
            "violations": violations,
            "new_greeks": new_greeks,
            "risk_score": self._calculate_risk_score(new_greeks)
        }

    def _calculate_risk_score(self, greeks: Dict[str, float]) -> float:
        """Calculate overall risk score (0-100)."""
        # Normalize each Greek to 0-100 scale
        delta_score = min(100, abs(greeks["delta_dollars"]) / (self.portfolio_value * 0.20) * 100)
        theta_score = min(100, abs(greeks["theta"]) / abs(self.max_portfolio_theta) * 100)
        vega_score = min(100, abs(greeks["vega"] * 10) / (self.portfolio_value * 0.05) * 100)

        # Weighted average
        return (delta_score * 0.4 + theta_score * 0.3 + vega_score * 0.3)


# Example usage and testing
if __name__ == "__main__":
    print("Options Strategy Playbook - Testing Framework")
    print("=" * 80)

    # Test Protective Put Strategy
    print("\n1. PROTECTIVE PUT STRATEGY TEST")
    print("-" * 40)
    protective_puts = ProtectivePutStrategy(portfolio_value=1000000, max_insurance_cost=0.02)

    # Test deployment conditions
    deployment = protective_puts.determine_deployment_conditions(
        vix=28, market_drawdown=7, earnings_days=10, fed_days=20
    )
    print(f"Deploy Protective Puts: {deployment['deploy']}")
    print(f"Urgency: {deployment['urgency']}")
    print(f"Reasoning: {deployment['reasoning']}")

    # Test strike selection
    strike_info = protective_puts.select_strike(
        current_price=100, volatility=0.30, protection_level="standard"
    )
    print(f"Selected Strike: ${strike_info['strike']:.2f}")
    print(f"Protection starts at: ${strike_info['protection_starts_at']:.2f}")

    # Test Covered Call Strategy
    print("\n2. COVERED CALL STRATEGY TEST")
    print("-" * 40)
    covered_calls = CoveredCallStrategy(position_size=1000, target_monthly_income=0.02)

    # Test strike selection
    cc_strike = covered_calls.select_strike(
        current_price=100, volatility=0.25, technical_resistance=105, iv_rank=60
    )
    print(f"Selected Strike: ${cc_strike['strike']:.2f}")
    print(f"Annualized Return: {cc_strike['annualized_return']:.1f}%")
    print(f"Assignment Probability: {cc_strike['assignment_probability']:.1f}%")

    # Test Volatility Arbitrage
    print("\n3. VOLATILITY ARBITRAGE TEST")
    print("-" * 40)
    vol_arb = VolatilityArbitrageStrategy(capital_allocated=100000, max_vega_exposure=10000)

    # Test IV/RV divergence
    iv_rv = vol_arb.identify_iv_rv_divergence(
        symbol="SPY", iv_30d=0.25, rv_30d=0.18, rv_60d=0.20
    )
    print(f"Signal: {iv_rv['signal']}")
    print(f"Strategy: {iv_rv['strategy']}")
    print(f"Confidence: {iv_rv['confidence']:.0f}%")

    # Test Earnings Strategy
    print("\n4. EARNINGS STRATEGY TEST")
    print("-" * 40)
    earnings = EarningsStrategy(capital_per_trade=5000)

    # Test earnings analysis
    earnings_setup = earnings.analyze_earnings_setup(
        symbol="AAPL",
        days_to_earnings=5,
        iv_current=0.35,
        iv_historical=[0.25, 0.30, 0.28, 0.32],
        historical_moves=[0.04, 0.06, 0.05, 0.07]
    )
    print(f"Strategy: {earnings_setup['strategy']}")
    print(f"Confidence: {earnings_setup['confidence']}%")
    print(f"Expected Move: {earnings_setup['metrics']['expected_move_pct']:.1f}%")
    print(f"Reasoning: {earnings_setup['reasoning']}")

    print("\n" + "=" * 80)
    print("Options Playbook Testing Complete")