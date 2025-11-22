"""
Dynamic Risk Adjustment Framework
Automatically adjust risk parameters based on market conditions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from scipy.signal import find_peaks
import talib

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_QUIET = "bull_quiet"          # Low vol uptrend
    BULL_VOLATILE = "bull_volatile"    # High vol uptrend
    BEAR_QUIET = "bear_quiet"          # Low vol downtrend
    BEAR_VOLATILE = "bear_volatile"    # High vol downtrend
    RANGING = "ranging"                # Sideways market
    CRISIS = "crisis"                  # Extreme conditions


@dataclass
class RiskAdjustmentRules:
    """Rules for dynamic risk adjustment"""

    # Position sizing adjustments by regime
    position_size_multipliers = {
        MarketRegime.BULL_QUIET: 1.2,      # Increase size in calm bulls
        MarketRegime.BULL_VOLATILE: 0.9,   # Slight reduction in volatile bulls
        MarketRegime.BEAR_QUIET: 0.7,      # Reduce in bears
        MarketRegime.BEAR_VOLATILE: 0.5,   # Significant reduction
        MarketRegime.RANGING: 0.8,         # Moderate size in ranging
        MarketRegime.CRISIS: 0.3           # Minimum size in crisis
    }

    # Stop loss adjustments (tighter in volatile markets)
    stop_loss_multipliers = {
        MarketRegime.BULL_QUIET: 1.5,      # Wider stops in calm
        MarketRegime.BULL_VOLATILE: 1.0,   # Normal stops
        MarketRegime.BEAR_QUIET: 0.8,      # Tighter stops
        MarketRegime.BEAR_VOLATILE: 0.6,   # Very tight stops
        MarketRegime.RANGING: 1.0,         # Normal stops
        MarketRegime.CRISIS: 0.5           # Extremely tight
    }

    # Cash allocation by regime
    target_cash_allocation = {
        MarketRegime.BULL_QUIET: 0.05,     # 5% cash
        MarketRegime.BULL_VOLATILE: 0.10,  # 10% cash
        MarketRegime.BEAR_QUIET: 0.20,     # 20% cash
        MarketRegime.BEAR_VOLATILE: 0.30,  # 30% cash
        MarketRegime.RANGING: 0.15,        # 15% cash
        MarketRegime.CRISIS: 0.40          # 40% cash
    }

    # VIX thresholds for regime changes
    vix_thresholds = {
        'low': 15,          # Below = quiet
        'medium': 25,       # Below = normal
        'high': 35,         # Below = volatile
        'extreme': 45       # Above = crisis
    }


class VolatilityRegimeDetector:
    """
    Detect volatility regimes and market conditions
    """

    def __init__(self, lookback_period: int = 60):
        """
        Initialize volatility regime detector

        Args:
            lookback_period: Days to look back for regime detection
        """
        self.lookback_period = lookback_period
        self.rules = RiskAdjustmentRules()

    def detect_regime(self,
                     market_data: pd.DataFrame,
                     vix: float) -> MarketRegime:
        """
        Detect current market regime

        Args:
            market_data: DataFrame with OHLCV data
            vix: Current VIX level

        Returns:
            Current MarketRegime
        """
        if len(market_data) < self.lookback_period:
            logger.warning("Insufficient data for regime detection, defaulting to RANGING")
            return MarketRegime.RANGING

        # Calculate returns
        returns = market_data['close'].pct_change().dropna()

        # Determine trend (bull/bear/ranging)
        trend = self._detect_trend(market_data)

        # Determine volatility level
        volatility_level = self._classify_volatility(vix, returns)

        # Combine to determine regime
        if volatility_level == 'crisis':
            return MarketRegime.CRISIS

        if trend == 'bull':
            if volatility_level == 'low':
                return MarketRegime.BULL_QUIET
            else:
                return MarketRegime.BULL_VOLATILE
        elif trend == 'bear':
            if volatility_level == 'low':
                return MarketRegime.BEAR_QUIET
            else:
                return MarketRegime.BEAR_VOLATILE
        else:
            return MarketRegime.RANGING

    def _detect_trend(self, market_data: pd.DataFrame) -> str:
        """
        Detect market trend using multiple indicators

        Returns: 'bull', 'bear', or 'ranging'
        """
        close = market_data['close'].values

        # Moving averages
        sma_20 = talib.SMA(close, timeperiod=20)[-1]
        sma_50 = talib.SMA(close, timeperiod=50)[-1]
        current_price = close[-1]

        # Trend strength using ADX
        high = market_data['high'].values
        low = market_data['low'].values
        adx = talib.ADX(high, low, close, timeperiod=14)[-1]

        # Price position relative to moving averages
        above_short = current_price > sma_20
        above_long = current_price > sma_50
        ma_aligned = sma_20 > sma_50

        # Determine trend
        if adx < 25:  # Weak trend
            return 'ranging'
        elif above_short and above_long and ma_aligned:
            return 'bull'
        elif not above_short and not above_long and not ma_aligned:
            return 'bear'
        else:
            return 'ranging'

    def _classify_volatility(self, vix: float, returns: pd.Series) -> str:
        """
        Classify volatility level

        Returns: 'low', 'medium', 'high', or 'crisis'
        """
        thresholds = self.rules.vix_thresholds

        # Primary classification based on VIX
        if vix >= thresholds['extreme']:
            return 'crisis'
        elif vix >= thresholds['high']:
            return 'high'
        elif vix >= thresholds['medium']:
            return 'medium'
        elif vix >= thresholds['low']:
            return 'low'
        else:
            # Very low VIX, check realized volatility
            realized_vol = returns.std() * np.sqrt(252)
            if realized_vol > 0.30:  # 30% annualized vol
                return 'high'
            elif realized_vol > 0.20:
                return 'medium'
            else:
                return 'low'

    def calculate_regime_probability(self,
                                    market_data: pd.DataFrame,
                                    n_periods: int = 20) -> Dict[MarketRegime, float]:
        """
        Calculate probability of each regime over recent periods

        Useful for gradual transitions between regimes
        """
        if len(market_data) < n_periods + self.lookback_period:
            # Not enough data, return equal probabilities
            regimes = list(MarketRegime)
            return {regime: 1.0 / len(regimes) for regime in regimes}

        regime_counts = {regime: 0 for regime in MarketRegime}

        # Sample different periods
        for i in range(n_periods):
            sample_data = market_data.iloc[-(self.lookback_period + i):len(market_data) - i]
            if len(sample_data) >= self.lookback_period:
                # Estimate VIX from realized volatility if not available
                returns = sample_data['close'].pct_change().dropna()
                realized_vol = returns.std() * np.sqrt(252)
                estimated_vix = realized_vol * 100  # Rough conversion

                regime = self.detect_regime(sample_data, estimated_vix)
                regime_counts[regime] += 1

        # Convert to probabilities
        total = sum(regime_counts.values())
        if total > 0:
            return {regime: count / total for regime, count in regime_counts.items()}
        else:
            regimes = list(MarketRegime)
            return {regime: 1.0 / len(regimes) for regime in regimes}


class DynamicRiskAdjuster:
    """
    Main dynamic risk adjustment engine
    """

    def __init__(self):
        self.regime_detector = VolatilityRegimeDetector()
        self.rules = RiskAdjustmentRules()
        self.current_regime = MarketRegime.RANGING
        self.regime_history = []
        self.adjustment_history = []

    def calculate_risk_adjustments(self,
                                  market_data: pd.DataFrame,
                                  vix: float,
                                  current_positions: Dict[str, float],
                                  account_value: float) -> Dict:
        """
        Calculate all risk adjustments based on current market conditions

        Returns:
            Dictionary with all adjustment parameters
        """
        # Detect current regime
        self.current_regime = self.regime_detector.detect_regime(market_data, vix)
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': self.current_regime
        })

        # Calculate regime probabilities for smooth transitions
        regime_probs = self.regime_detector.calculate_regime_probability(market_data)

        # Get base adjustments for current regime
        position_multiplier = self.rules.position_size_multipliers[self.current_regime]
        stop_loss_multiplier = self.rules.stop_loss_multipliers[self.current_regime]
        target_cash = self.rules.target_cash_allocation[self.current_regime]

        # Apply probability weighting for smoother transitions
        weighted_position_mult = sum(
            prob * self.rules.position_size_multipliers[regime]
            for regime, prob in regime_probs.items()
        )
        weighted_stop_mult = sum(
            prob * self.rules.stop_loss_multipliers[regime]
            for regime, prob in regime_probs.items()
        )
        weighted_cash = sum(
            prob * self.rules.target_cash_allocation[regime]
            for regime, prob in regime_probs.items()
        )

        # Calculate specific adjustments
        adjustments = {
            'timestamp': datetime.now(),
            'current_regime': self.current_regime.value,
            'regime_probabilities': {r.value: p for r, p in regime_probs.items()},
            'position_size_adjustment': {
                'base_multiplier': position_multiplier,
                'weighted_multiplier': weighted_position_mult,
                'recommendation': self._get_position_recommendation(weighted_position_mult)
            },
            'stop_loss_adjustment': {
                'base_multiplier': stop_loss_multiplier,
                'weighted_multiplier': weighted_stop_mult,
                'recommendation': self._get_stop_loss_recommendation(weighted_stop_mult)
            },
            'cash_allocation': {
                'current_cash': self._calculate_current_cash_percentage(current_positions, account_value),
                'target_cash': target_cash,
                'weighted_target': weighted_cash,
                'action_required': self._get_cash_action(current_positions, account_value, weighted_cash)
            },
            'vix_level': vix,
            'risk_score': self._calculate_risk_score(vix, self.current_regime),
            'recommendations': []
        }

        # Generate specific recommendations
        adjustments['recommendations'] = self._generate_recommendations(
            adjustments,
            current_positions,
            account_value
        )

        # Store adjustment history
        self.adjustment_history.append(adjustments)

        return adjustments

    def _get_position_recommendation(self, multiplier: float) -> str:
        """Generate position sizing recommendation"""
        if multiplier >= 1.1:
            return "Increase position sizes by {:.0%}".format(multiplier - 1)
        elif multiplier <= 0.6:
            return "Significantly reduce position sizes by {:.0%}".format(1 - multiplier)
        elif multiplier <= 0.9:
            return "Moderately reduce position sizes by {:.0%}".format(1 - multiplier)
        else:
            return "Maintain current position sizes"

    def _get_stop_loss_recommendation(self, multiplier: float) -> str:
        """Generate stop loss recommendation"""
        if multiplier <= 0.6:
            return "Tighten stop losses to {:.0%} of normal".format(multiplier)
        elif multiplier <= 0.9:
            return "Moderately tighten stop losses"
        elif multiplier >= 1.3:
            return "Consider wider stop losses for reduced whipsaws"
        else:
            return "Maintain current stop loss levels"

    def _calculate_current_cash_percentage(self,
                                          positions: Dict[str, float],
                                          account_value: float) -> float:
        """Calculate current cash allocation percentage"""
        total_positions = sum(positions.values())
        cash = account_value - total_positions
        return cash / account_value if account_value > 0 else 0

    def _get_cash_action(self,
                        positions: Dict[str, float],
                        account_value: float,
                        target_cash: float) -> str:
        """Determine required cash allocation action"""
        current_cash_pct = self._calculate_current_cash_percentage(positions, account_value)
        difference = target_cash - current_cash_pct

        if abs(difference) < 0.02:  # Within 2%
            return "Cash allocation optimal"
        elif difference > 0:
            return f"Increase cash by {difference:.1%} (sell positions)"
        else:
            return f"Deploy {-difference:.1%} of cash"

    def _calculate_risk_score(self, vix: float, regime: MarketRegime) -> float:
        """
        Calculate overall risk score (0-100)
        Higher = riskier environment
        """
        # VIX component (0-50 points)
        vix_score = min(vix, 50)

        # Regime component (0-50 points)
        regime_scores = {
            MarketRegime.BULL_QUIET: 10,
            MarketRegime.RANGING: 20,
            MarketRegime.BULL_VOLATILE: 30,
            MarketRegime.BEAR_QUIET: 35,
            MarketRegime.BEAR_VOLATILE: 45,
            MarketRegime.CRISIS: 50
        }
        regime_score = regime_scores[regime]

        return vix_score + regime_score

    def _generate_recommendations(self,
                                 adjustments: Dict,
                                 positions: Dict[str, float],
                                 account_value: float) -> List[str]:
        """Generate specific action recommendations"""
        recommendations = []
        risk_score = adjustments['risk_score']

        # High risk recommendations
        if risk_score > 70:
            recommendations.append("⚠️ HIGH RISK: Consider defensive positioning")
            recommendations.append("Reduce leverage and high-beta positions")
            recommendations.append("Increase cash reserves to minimum 30%")
            recommendations.append("Implement tight stop losses on all positions")

        # Medium risk recommendations
        elif risk_score > 40:
            recommendations.append("MODERATE RISK: Proceed with caution")
            recommendations.append("Review and tighten stop losses")
            recommendations.append("Consider reducing position sizes by 20-30%")
            recommendations.append("Avoid new aggressive positions")

        # Low risk recommendations
        else:
            recommendations.append("LOW RISK: Favorable conditions for trading")
            recommendations.append("Consider increasing position sizes on high-conviction trades")
            recommendations.append("Opportunity to deploy excess cash reserves")

        # Specific position recommendations
        if adjustments['position_size_adjustment']['weighted_multiplier'] < 0.7:
            recommendations.append(f"Reduce all position sizes to {adjustments['position_size_adjustment']['weighted_multiplier']:.0%} of normal")

        # Cash rebalancing
        cash_action = adjustments['cash_allocation']['action_required']
        if 'Increase cash' in cash_action or 'sell positions' in cash_action:
            recommendations.append(f"Action Required: {cash_action}")

        return recommendations


class MarketConditionAnalyzer:
    """
    Analyze broader market conditions for risk adjustment
    """

    def __init__(self):
        self.indicators = {}

    def analyze_market_breadth(self,
                              advance_decline_data: pd.DataFrame) -> Dict:
        """
        Analyze market breadth indicators

        Args:
            advance_decline_data: DataFrame with advancing/declining stocks
        """
        if advance_decline_data.empty:
            return {'breadth_score': 50, 'signal': 'neutral'}

        # Calculate advance/decline line
        advances = advance_decline_data['advances'].values
        declines = advance_decline_data['declines'].values
        ad_line = np.cumsum(advances - declines)

        # Calculate breadth indicators
        advance_ratio = advances[-1] / (advances[-1] + declines[-1]) if (advances[-1] + declines[-1]) > 0 else 0.5

        # Moving average of A/D line
        ad_ma = pd.Series(ad_line).rolling(10).mean().iloc[-1]
        ad_trend = 'positive' if ad_line[-1] > ad_ma else 'negative'

        # Calculate McClellan Oscillator (simplified)
        ema_19 = pd.Series(advances - declines).ewm(span=19).mean().iloc[-1]
        ema_39 = pd.Series(advances - declines).ewm(span=39).mean().iloc[-1]
        mcclellan = ema_19 - ema_39

        # Determine signal
        if advance_ratio > 0.65 and ad_trend == 'positive':
            signal = 'bullish'
            score = 70 + (advance_ratio - 0.65) * 100
        elif advance_ratio < 0.35 and ad_trend == 'negative':
            signal = 'bearish'
            score = 30 - (0.35 - advance_ratio) * 100
        else:
            signal = 'neutral'
            score = 50

        return {
            'breadth_score': max(0, min(100, score)),
            'signal': signal,
            'advance_ratio': advance_ratio,
            'ad_trend': ad_trend,
            'mcclellan_oscillator': mcclellan
        }

    def analyze_sector_rotation(self,
                               sector_returns: Dict[str, float]) -> Dict:
        """
        Analyze sector rotation for risk signals

        Args:
            sector_returns: Dictionary of sector -> recent returns
        """
        if not sector_returns:
            return {'rotation_signal': 'unknown', 'risk_sectors': []}

        # Defensive sectors
        defensive = ['Utilities', 'Consumer Staples', 'Healthcare']
        cyclical = ['Technology', 'Consumer Discretionary', 'Financials']

        defensive_avg = np.mean([sector_returns.get(s, 0) for s in defensive])
        cyclical_avg = np.mean([sector_returns.get(s, 0) for s in cyclical])

        # Identify rotation
        if defensive_avg > cyclical_avg * 1.2:
            rotation_signal = 'defensive'
            risk_indication = 'risk_off'
        elif cyclical_avg > defensive_avg * 1.2:
            rotation_signal = 'cyclical'
            risk_indication = 'risk_on'
        else:
            rotation_signal = 'mixed'
            risk_indication = 'neutral'

        # Identify weakest sectors (potential risks)
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1])
        risk_sectors = [s[0] for s in sorted_sectors[:3] if s[1] < -0.05]

        return {
            'rotation_signal': rotation_signal,
            'risk_indication': risk_indication,
            'defensive_performance': defensive_avg,
            'cyclical_performance': cyclical_avg,
            'risk_sectors': risk_sectors
        }


class AdaptiveStopLossManager:
    """
    Dynamically adjust stop losses based on market conditions
    """

    def __init__(self):
        self.atr_multiplier_base = 2.0  # Base ATR multiplier for stops

    def calculate_adaptive_stop_loss(self,
                                    entry_price: float,
                                    current_price: float,
                                    atr: float,
                                    volatility_regime: str,
                                    position_type: str = 'long') -> Dict:
        """
        Calculate adaptive stop loss

        Args:
            entry_price: Position entry price
            current_price: Current market price
            atr: Average True Range
            volatility_regime: Current volatility regime
            position_type: 'long' or 'short'

        Returns:
            Dictionary with stop loss parameters
        """
        # Adjust ATR multiplier based on regime
        regime_multipliers = {
            'low': 2.5,      # Wider stops in low vol
            'medium': 2.0,   # Normal stops
            'high': 1.5,     # Tighter stops in high vol
            'crisis': 1.0    # Very tight in crisis
        }

        atr_multiplier = regime_multipliers.get(volatility_regime, 2.0)

        # Calculate stop distance
        stop_distance = atr * atr_multiplier

        # Calculate stop levels
        if position_type == 'long':
            # Initial stop
            initial_stop = entry_price - stop_distance

            # Trailing stop (if position is profitable)
            if current_price > entry_price:
                trailing_stop = current_price - stop_distance
                recommended_stop = max(initial_stop, trailing_stop)

                # Breakeven stop (if significantly profitable)
                if current_price > entry_price * 1.03:  # 3% profit
                    breakeven_stop = entry_price * 1.001  # Small buffer above entry
                    recommended_stop = max(recommended_stop, breakeven_stop)
            else:
                recommended_stop = initial_stop

        else:  # short position
            # Initial stop
            initial_stop = entry_price + stop_distance

            # Trailing stop (if position is profitable)
            if current_price < entry_price:
                trailing_stop = current_price + stop_distance
                recommended_stop = min(initial_stop, trailing_stop)

                # Breakeven stop (if significantly profitable)
                if current_price < entry_price * 0.97:  # 3% profit
                    breakeven_stop = entry_price * 0.999  # Small buffer below entry
                    recommended_stop = min(recommended_stop, breakeven_stop)
            else:
                recommended_stop = initial_stop

        # Calculate risk metrics
        stop_distance_pct = abs(recommended_stop - current_price) / current_price
        potential_loss = abs(recommended_stop - entry_price) / entry_price

        return {
            'recommended_stop': recommended_stop,
            'stop_distance': stop_distance,
            'stop_distance_pct': stop_distance_pct,
            'atr_multiplier': atr_multiplier,
            'potential_loss_pct': potential_loss,
            'stop_type': 'trailing' if current_price > entry_price else 'fixed',
            'volatility_regime': volatility_regime
        }

    def calculate_time_based_stop_adjustment(self,
                                            days_in_position: int,
                                            initial_stop: float,
                                            current_price: float,
                                            position_type: str = 'long') -> float:
        """
        Adjust stop loss based on time in position

        Gradually tighten stops for positions held too long
        """
        # Time decay factor (tighten stops over time)
        if days_in_position < 5:
            time_factor = 1.0  # No adjustment
        elif days_in_position < 20:
            time_factor = 0.95  # 5% tighter
        elif days_in_position < 60:
            time_factor = 0.90  # 10% tighter
        else:
            time_factor = 0.85  # 15% tighter

        # Calculate adjusted stop
        if position_type == 'long':
            stop_distance = current_price - initial_stop
            adjusted_stop = current_price - (stop_distance * time_factor)
        else:
            stop_distance = initial_stop - current_price
            adjusted_stop = current_price + (stop_distance * time_factor)

        return adjusted_stop