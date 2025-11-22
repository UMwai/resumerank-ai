"""
Position Sizing Framework
Implements Kelly Criterion, volatility-adjusted, and correlation-adjusted position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionLimits:
    """Position limits by strategy type"""
    max_single_position_pct: float = 0.15  # Max 15% in any single position
    max_sector_exposure_pct: float = 0.35   # Max 35% in any sector
    max_correlated_exposure: float = 0.25   # Max 25% in highly correlated assets

    # Strategy-specific limits
    momentum_max_position: float = 0.10     # Momentum strategies: max 10%
    mean_reversion_max_position: float = 0.08  # Mean reversion: max 8%
    event_driven_max_position: float = 0.12    # Event-driven: max 12%
    arbitrage_max_position: float = 0.15       # Arbitrage: max 15%

    # Risk tier limits (based on volatility percentile)
    low_risk_max: float = 0.15      # 0-33rd percentile volatility
    medium_risk_max: float = 0.10   # 33-66th percentile
    high_risk_max: float = 0.05     # 66-100th percentile


class KellyCriterion:
    """
    Kelly Criterion implementation for optimal position sizing
    Includes both unconstrained and constrained versions
    """

    def __init__(self, kelly_fraction: float = 0.25):
        """
        Initialize Kelly Criterion calculator

        Args:
            kelly_fraction: Fraction of Kelly to use (default 0.25 for 1/4 Kelly)
        """
        self.kelly_fraction = kelly_fraction
        self.min_position = 0.001  # Minimum 0.1% position
        self.max_position = 0.25   # Maximum 25% position

    def calculate_kelly_unconstrained(self,
                                     win_prob: float,
                                     win_return: float,
                                     loss_return: float) -> float:
        """
        Calculate unconstrained Kelly fraction

        Formula: f* = (p * b - q) / b
        where:
            p = probability of winning
            q = probability of losing (1 - p)
            b = ratio of win to loss amounts
        """
        if loss_return >= 0:
            logger.warning("Loss return must be negative")
            return 0.0

        q = 1 - win_prob
        b = abs(win_return / loss_return)

        kelly = (win_prob * b - q) / b

        # Ensure positive allocation only when expected value is positive
        if kelly <= 0:
            return 0.0

        return kelly

    def calculate_kelly_constrained(self,
                                  win_prob: float,
                                  win_return: float,
                                  loss_return: float,
                                  confidence: float = 0.95) -> float:
        """
        Calculate constrained Kelly fraction with confidence adjustment

        Adjusts for parameter uncertainty using confidence intervals
        """
        # Base Kelly calculation
        base_kelly = self.calculate_kelly_unconstrained(win_prob, win_return, loss_return)

        # Adjust for parameter uncertainty
        # Reduce Kelly based on confidence in estimates
        uncertainty_adjustment = confidence
        adjusted_kelly = base_kelly * uncertainty_adjustment * self.kelly_fraction

        # Apply position limits
        return np.clip(adjusted_kelly, self.min_position, self.max_position)

    def calculate_multi_asset_kelly(self,
                                   expected_returns: np.ndarray,
                                   covariance_matrix: np.ndarray,
                                   risk_free_rate: float = 0.02) -> np.ndarray:
        """
        Calculate Kelly weights for multiple assets using mean-variance optimization

        Solves: max(w' * μ - λ * w' * Σ * w)
        where λ = 1 / (2 * risk_tolerance)
        """
        n_assets = len(expected_returns)

        # Excess returns
        excess_returns = expected_returns - risk_free_rate

        # Kelly optimal weights (unconstrained)
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            kelly_weights = inv_cov @ excess_returns

            # Normalize to sum to 1 (fully invested)
            kelly_weights = kelly_weights / np.sum(np.abs(kelly_weights))

            # Apply fractional Kelly
            kelly_weights *= self.kelly_fraction

            # Apply individual position constraints
            kelly_weights = np.clip(kelly_weights, -self.max_position, self.max_position)

            return kelly_weights

        except np.linalg.LinAlgError:
            logger.error("Covariance matrix is singular, returning equal weights")
            return np.ones(n_assets) / n_assets


class VolatilityAdjustedSizing:
    """
    Volatility-adjusted position sizing
    Scales positions inversely with volatility to maintain constant risk
    """

    def __init__(self, target_volatility: float = 0.15):
        """
        Initialize volatility-adjusted sizing

        Args:
            target_volatility: Annual target volatility (default 15%)
        """
        self.target_volatility = target_volatility
        self.lookback_period = 20  # Days for volatility calculation
        self.ewm_span = 10  # Exponential weighting span

    def calculate_position_size(self,
                               historical_returns: pd.Series,
                               base_position: float,
                               use_ewm: bool = True) -> float:
        """
        Calculate volatility-adjusted position size

        Formula: Position Size = (Target Vol / Realized Vol) * Base Position
        """
        if len(historical_returns) < self.lookback_period:
            logger.warning(f"Insufficient data for volatility calculation, using base position")
            return base_position

        # Calculate realized volatility
        if use_ewm:
            # Exponentially weighted volatility (more responsive to recent changes)
            realized_vol = historical_returns.ewm(span=self.ewm_span).std().iloc[-1]
        else:
            # Simple rolling volatility
            realized_vol = historical_returns.rolling(self.lookback_period).std().iloc[-1]

        # Annualize volatility
        realized_vol_annual = realized_vol * np.sqrt(252)

        # Avoid division by zero
        if realized_vol_annual < 0.001:
            realized_vol_annual = 0.001

        # Calculate adjustment factor
        vol_adjustment = min(self.target_volatility / realized_vol_annual, 2.0)  # Cap at 2x

        # Apply adjustment
        adjusted_position = base_position * vol_adjustment

        return adjusted_position

    def calculate_atr_position_size(self,
                                   account_value: float,
                                   atr: float,
                                   risk_per_trade: float = 0.01) -> float:
        """
        Calculate position size using Average True Range (ATR)

        Formula: Position Size = (Account Value * Risk%) / (ATR * ATR Multiplier)
        """
        atr_multiplier = 2.0  # Stop loss at 2 ATRs

        if atr <= 0:
            logger.error("ATR must be positive")
            return 0.0

        position_value = (account_value * risk_per_trade) / (atr * atr_multiplier)

        # Apply maximum position limit
        max_position_value = account_value * 0.15

        return min(position_value, max_position_value)


class CorrelationAdjustedSizing:
    """
    Adjust position sizes based on portfolio correlations
    Reduces positions when assets are highly correlated
    """

    def __init__(self, correlation_threshold: float = 0.7):
        """
        Initialize correlation-adjusted sizing

        Args:
            correlation_threshold: Threshold for high correlation (default 0.7)
        """
        self.correlation_threshold = correlation_threshold
        self.lookback_period = 60  # Days for correlation calculation

    def calculate_correlation_matrix(self,
                                    returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlation matrix"""
        return returns_df.rolling(self.lookback_period).corr()

    def adjust_positions_for_correlation(self,
                                        positions: Dict[str, float],
                                        correlation_matrix: pd.DataFrame,
                                        max_correlated_exposure: float = 0.25) -> Dict[str, float]:
        """
        Adjust position sizes based on correlations

        Reduces positions in highly correlated assets to limit concentration risk
        """
        adjusted_positions = positions.copy()
        symbols = list(positions.keys())

        # Identify highly correlated pairs
        high_correlation_pairs = []
        for i, sym1 in enumerate(symbols):
            for j, sym2 in enumerate(symbols[i+1:], i+1):
                if sym1 in correlation_matrix.index and sym2 in correlation_matrix.columns:
                    corr = correlation_matrix.loc[sym1, sym2]
                    if abs(corr) > self.correlation_threshold:
                        high_correlation_pairs.append((sym1, sym2, corr))

        # Group correlated assets into clusters
        correlation_clusters = self._find_correlation_clusters(symbols, high_correlation_pairs)

        # Adjust positions for each cluster
        for cluster in correlation_clusters:
            if len(cluster) > 1:
                # Calculate total exposure in cluster
                cluster_exposure = sum(positions[sym] for sym in cluster if sym in positions)

                if cluster_exposure > max_correlated_exposure:
                    # Scale down all positions in cluster proportionally
                    scale_factor = max_correlated_exposure / cluster_exposure
                    for sym in cluster:
                        if sym in adjusted_positions:
                            adjusted_positions[sym] *= scale_factor
                            logger.info(f"Reduced position in {sym} due to correlation: "
                                      f"{positions[sym]:.2%} -> {adjusted_positions[sym]:.2%}")

        return adjusted_positions

    def _find_correlation_clusters(self,
                                  symbols: List[str],
                                  correlation_pairs: List[Tuple[str, str, float]]) -> List[List[str]]:
        """Find clusters of correlated assets using union-find algorithm"""
        parent = {sym: sym for sym in symbols}

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union correlated pairs
        for sym1, sym2, _ in correlation_pairs:
            union(sym1, sym2)

        # Group by cluster
        clusters = {}
        for sym in symbols:
            root = find(sym)
            if root not in clusters:
                clusters[root] = []
            clusters[root].append(sym)

        return list(clusters.values())


class PositionSizeCalculator:
    """
    Main position size calculator that combines all sizing methods
    """

    def __init__(self,
                 account_value: float,
                 max_portfolio_risk: float = 0.06,
                 position_limits: Optional[PositionLimits] = None):
        """
        Initialize position size calculator

        Args:
            account_value: Total account value
            max_portfolio_risk: Maximum portfolio risk (default 6%)
            position_limits: Position limit configuration
        """
        self.account_value = account_value
        self.max_portfolio_risk = max_portfolio_risk
        self.position_limits = position_limits or PositionLimits()

        self.kelly = KellyCriterion(kelly_fraction=0.25)
        self.vol_sizing = VolatilityAdjustedSizing(target_volatility=0.15)
        self.corr_sizing = CorrelationAdjustedSizing(correlation_threshold=0.7)

    def calculate_optimal_position_size(self,
                                       strategy_type: str,
                                       expected_return: float,
                                       risk_metrics: Dict,
                                       current_positions: Dict[str, float],
                                       correlation_matrix: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate optimal position size combining all methods

        Returns dictionary with:
        - base_size: Initial position size
        - kelly_size: Kelly criterion size
        - vol_adjusted_size: Volatility-adjusted size
        - corr_adjusted_size: Correlation-adjusted size
        - final_size: Final recommended size
        - risk_metrics: Associated risk metrics
        """
        result = {}

        # 1. Calculate base Kelly position
        win_prob = risk_metrics.get('win_probability', 0.55)
        avg_win = risk_metrics.get('avg_win_return', 0.05)
        avg_loss = risk_metrics.get('avg_loss_return', -0.03)

        kelly_size = self.kelly.calculate_kelly_constrained(
            win_prob, avg_win, avg_loss,
            confidence=risk_metrics.get('confidence', 0.85)
        )
        result['kelly_size'] = kelly_size

        # 2. Apply strategy-specific limits
        strategy_limits = {
            'momentum': self.position_limits.momentum_max_position,
            'mean_reversion': self.position_limits.mean_reversion_max_position,
            'event_driven': self.position_limits.event_driven_max_position,
            'arbitrage': self.position_limits.arbitrage_max_position
        }

        max_strategy_size = strategy_limits.get(strategy_type,
                                                self.position_limits.max_single_position_pct)

        # 3. Volatility adjustment
        if 'historical_returns' in risk_metrics:
            vol_adjusted = self.vol_sizing.calculate_position_size(
                risk_metrics['historical_returns'],
                min(kelly_size, max_strategy_size)
            )
            result['vol_adjusted_size'] = vol_adjusted
        else:
            vol_adjusted = min(kelly_size, max_strategy_size)
            result['vol_adjusted_size'] = vol_adjusted

        # 4. Risk tier adjustment based on volatility percentile
        vol_percentile = risk_metrics.get('volatility_percentile', 50)
        if vol_percentile < 33:
            risk_tier_limit = self.position_limits.low_risk_max
        elif vol_percentile < 66:
            risk_tier_limit = self.position_limits.medium_risk_max
        else:
            risk_tier_limit = self.position_limits.high_risk_max

        # 5. Correlation adjustment if portfolio exists
        size_before_corr = min(vol_adjusted, risk_tier_limit)

        if correlation_matrix is not None and len(current_positions) > 0:
            # Add new position temporarily for correlation check
            temp_positions = current_positions.copy()
            temp_positions[risk_metrics.get('symbol', 'NEW')] = size_before_corr

            adjusted_positions = self.corr_sizing.adjust_positions_for_correlation(
                temp_positions,
                correlation_matrix,
                self.position_limits.max_correlated_exposure
            )

            final_size = adjusted_positions[risk_metrics.get('symbol', 'NEW')]
            result['corr_adjusted_size'] = final_size
        else:
            final_size = size_before_corr
            result['corr_adjusted_size'] = final_size

        # 6. Final safety checks
        # Check total portfolio risk
        total_positions = sum(current_positions.values())
        if total_positions + final_size > 1.0:
            # Scale down to maintain full investment constraint
            final_size = max(0, 1.0 - total_positions)

        # Ensure minimum position size for viability
        if final_size < self.kelly.min_position:
            final_size = 0  # Don't take position if too small

        result['final_size'] = final_size

        # 7. Calculate risk metrics for the position
        position_value = self.account_value * final_size
        result['risk_metrics'] = {
            'position_value': position_value,
            'max_loss': position_value * abs(avg_loss),
            'expected_return': position_value * expected_return,
            'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'portfolio_impact': final_size,
            'strategy_type': strategy_type,
            'volatility_tier': 'low' if vol_percentile < 33 else 'medium' if vol_percentile < 66 else 'high'
        }

        return result