"""
Portfolio-Level Risk Controls
Implements VaR, CVaR, concentration limits, leverage control, and cash management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PortfolioLimits:
    """Portfolio-level risk limits and thresholds"""
    # VaR/CVaR Limits
    var_95_limit: float = 0.05      # 5% max daily VaR at 95% confidence
    var_99_limit: float = 0.08      # 8% max daily VaR at 99% confidence
    cvar_95_limit: float = 0.07     # 7% max daily CVaR at 95% confidence
    cvar_99_limit: float = 0.10     # 10% max daily CVaR at 99% confidence

    # Concentration Limits
    max_sector_concentration: float = 0.35   # Max 35% in any sector
    max_stock_concentration: float = 0.15    # Max 15% in any single stock
    max_correlated_group: float = 0.40      # Max 40% in correlated assets (>0.6 correlation)

    # Leverage Limits
    max_gross_leverage: float = 1.5         # Max 150% gross exposure
    max_net_leverage: float = 1.0           # Max 100% net exposure
    initial_margin_requirement: float = 0.5  # 50% initial margin
    maintenance_margin: float = 0.25         # 25% maintenance margin

    # Cash Reserves
    min_cash_reserve_normal: float = 0.05   # 5% minimum cash in normal markets
    min_cash_reserve_volatile: float = 0.15  # 15% minimum cash in volatile markets
    min_cash_reserve_crisis: float = 0.30   # 30% minimum cash in crisis


class VaRMonitor:
    """
    Real-time Value at Risk monitoring
    Implements historical, parametric, and Monte Carlo VaR
    """

    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        """
        Initialize VaR Monitor

        Args:
            confidence_levels: List of confidence levels for VaR calculation
        """
        self.confidence_levels = confidence_levels
        self.lookback_period = 252  # Trading days for historical VaR
        self.decay_factor = 0.94     # EWMA decay factor

    def calculate_historical_var(self,
                                returns: pd.Series,
                                portfolio_value: float,
                                confidence_level: float = 0.95) -> float:
        """
        Calculate Historical VaR using percentile method

        Returns:
            VaR amount (positive number representing potential loss)
        """
        if len(returns) < 20:
            logger.warning("Insufficient data for historical VaR")
            return 0

        # Calculate percentile
        var_percentile = (1 - confidence_level) * 100
        var_return = np.percentile(returns, var_percentile)

        # Convert to dollar amount (return as positive loss)
        var_amount = -var_return * portfolio_value

        return max(0, var_amount)

    def calculate_parametric_var(self,
                                returns: pd.Series,
                                portfolio_value: float,
                                confidence_level: float = 0.95,
                                use_ewma: bool = True) -> float:
        """
        Calculate Parametric (Gaussian) VaR

        Assumes returns are normally distributed
        """
        if len(returns) < 20:
            logger.warning("Insufficient data for parametric VaR")
            return 0

        # Calculate mean and standard deviation
        mean_return = returns.mean()

        if use_ewma:
            # Exponentially weighted volatility
            std_return = returns.ewm(alpha=1-self.decay_factor).std().iloc[-1]
        else:
            std_return = returns.std()

        # Calculate z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence_level)

        # Calculate VaR
        var_return = mean_return + z_score * std_return
        var_amount = -var_return * portfolio_value

        return max(0, var_amount)

    def calculate_monte_carlo_var(self,
                                 returns: pd.Series,
                                 portfolio_value: float,
                                 confidence_level: float = 0.95,
                                 n_simulations: int = 10000,
                                 time_horizon: int = 1) -> float:
        """
        Calculate Monte Carlo VaR using simulated returns

        Args:
            returns: Historical returns
            portfolio_value: Current portfolio value
            confidence_level: Confidence level for VaR
            n_simulations: Number of Monte Carlo simulations
            time_horizon: Time horizon in days
        """
        if len(returns) < 60:
            logger.warning("Insufficient data for Monte Carlo VaR")
            return 0

        # Fit distribution parameters
        mean_return = returns.mean()
        std_return = returns.std()

        # Generate Monte Carlo simulations
        random_returns = np.random.normal(
            mean_return * time_horizon,
            std_return * np.sqrt(time_horizon),
            n_simulations
        )

        # Calculate portfolio values
        portfolio_values = portfolio_value * (1 + random_returns)

        # Calculate VaR
        var_percentile = (1 - confidence_level) * 100
        var_value = np.percentile(portfolio_values, var_percentile)
        var_amount = portfolio_value - var_value

        return max(0, var_amount)

    def calculate_combined_var(self,
                              returns: pd.Series,
                              portfolio_value: float,
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate VaR using all three methods and return combined results
        """
        results = {
            'historical': self.calculate_historical_var(returns, portfolio_value, confidence_level),
            'parametric': self.calculate_parametric_var(returns, portfolio_value, confidence_level),
            'monte_carlo': self.calculate_monte_carlo_var(returns, portfolio_value, confidence_level)
        }

        # Conservative approach: use maximum VaR
        results['combined'] = max(results.values())

        # Calculate as percentage of portfolio
        results['combined_pct'] = results['combined'] / portfolio_value if portfolio_value > 0 else 0

        return results


class CVaRCalculator:
    """
    Conditional Value at Risk (Expected Shortfall) Calculator
    Measures expected loss beyond VaR threshold
    """

    def __init__(self):
        self.var_monitor = VaRMonitor()

    def calculate_historical_cvar(self,
                                 returns: pd.Series,
                                 portfolio_value: float,
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate Historical CVaR (Expected Shortfall)

        Returns average loss beyond VaR threshold
        """
        if len(returns) < 20:
            logger.warning("Insufficient data for CVaR calculation")
            return 0

        # Find VaR threshold
        var_percentile = (1 - confidence_level) * 100
        var_threshold = np.percentile(returns, var_percentile)

        # Calculate average of returns below VaR threshold
        tail_returns = returns[returns <= var_threshold]

        if len(tail_returns) == 0:
            return 0

        cvar_return = tail_returns.mean()
        cvar_amount = -cvar_return * portfolio_value

        return max(0, cvar_amount)

    def calculate_parametric_cvar(self,
                                 returns: pd.Series,
                                 portfolio_value: float,
                                 confidence_level: float = 0.95) -> float:
        """
        Calculate Parametric CVaR assuming normal distribution

        Formula: CVaR = μ - σ * φ(z) / (1 - α)
        where φ is the standard normal PDF
        """
        if len(returns) < 20:
            logger.warning("Insufficient data for parametric CVaR")
            return 0

        mean_return = returns.mean()
        std_return = returns.std()

        # Calculate z-score
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)

        # Calculate CVaR using formula
        pdf_value = stats.norm.pdf(z_score)
        cvar_return = mean_return - std_return * (pdf_value / alpha)
        cvar_amount = -cvar_return * portfolio_value

        return max(0, cvar_amount)

    def calculate_combined_cvar(self,
                               returns: pd.Series,
                               portfolio_value: float,
                               confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Calculate CVaR using multiple methods
        """
        results = {
            'historical': self.calculate_historical_cvar(returns, portfolio_value, confidence_level),
            'parametric': self.calculate_parametric_cvar(returns, portfolio_value, confidence_level)
        }

        # Conservative approach: use maximum CVaR
        results['combined'] = max(results.values())
        results['combined_pct'] = results['combined'] / portfolio_value if portfolio_value > 0 else 0

        # Calculate CVaR/VaR ratio (should be > 1)
        var_results = self.var_monitor.calculate_combined_var(returns, portfolio_value, confidence_level)
        if var_results['combined'] > 0:
            results['cvar_var_ratio'] = results['combined'] / var_results['combined']
        else:
            results['cvar_var_ratio'] = 1.0

        return results


class ConcentrationLimits:
    """
    Monitor and enforce concentration limits at portfolio level
    """

    def __init__(self, limits: Optional[PortfolioLimits] = None):
        self.limits = limits or PortfolioLimits()

    def calculate_concentration_metrics(self,
                                       positions: Dict[str, float],
                                       sector_mapping: Dict[str, str]) -> Dict:
        """
        Calculate concentration metrics for portfolio

        Args:
            positions: Dictionary of symbol -> portfolio weight
            sector_mapping: Dictionary of symbol -> sector

        Returns:
            Dictionary with concentration metrics and violations
        """
        metrics = {
            'max_position': 0,
            'max_position_symbol': None,
            'sector_concentrations': {},
            'herfindahl_index': 0,
            'effective_n': 0,
            'violations': []
        }

        if not positions:
            return metrics

        # Calculate maximum position
        max_symbol = max(positions, key=positions.get)
        metrics['max_position'] = positions[max_symbol]
        metrics['max_position_symbol'] = max_symbol

        # Check single stock concentration limit
        if metrics['max_position'] > self.limits.max_stock_concentration:
            metrics['violations'].append({
                'type': 'stock_concentration',
                'symbol': max_symbol,
                'current': metrics['max_position'],
                'limit': self.limits.max_stock_concentration,
                'action': 'reduce_position'
            })

        # Calculate sector concentrations
        for symbol, weight in positions.items():
            sector = sector_mapping.get(symbol, 'Unknown')
            if sector not in metrics['sector_concentrations']:
                metrics['sector_concentrations'][sector] = 0
            metrics['sector_concentrations'][sector] += weight

        # Check sector concentration limits
        for sector, concentration in metrics['sector_concentrations'].items():
            if concentration > self.limits.max_sector_concentration:
                metrics['violations'].append({
                    'type': 'sector_concentration',
                    'sector': sector,
                    'current': concentration,
                    'limit': self.limits.max_sector_concentration,
                    'action': 'rebalance_sector'
                })

        # Calculate Herfindahl Index (measure of concentration)
        # HI = sum(wi^2), ranges from 1/n to 1
        metrics['herfindahl_index'] = sum(w**2 for w in positions.values())

        # Effective number of positions (1/HI)
        if metrics['herfindahl_index'] > 0:
            metrics['effective_n'] = 1 / metrics['herfindahl_index']
        else:
            metrics['effective_n'] = 0

        return metrics

    def check_correlation_concentration(self,
                                       positions: Dict[str, float],
                                       correlation_matrix: pd.DataFrame,
                                       correlation_threshold: float = 0.6) -> Dict:
        """
        Check concentration in correlated assets
        """
        results = {
            'correlated_groups': [],
            'max_correlated_exposure': 0,
            'violations': []
        }

        # Find groups of correlated assets
        symbols = list(positions.keys())
        processed = set()

        for symbol in symbols:
            if symbol in processed:
                continue

            # Find all assets correlated with this symbol
            correlated_group = [symbol]
            for other_symbol in symbols:
                if other_symbol != symbol and other_symbol not in processed:
                    if symbol in correlation_matrix.index and other_symbol in correlation_matrix.columns:
                        corr = correlation_matrix.loc[symbol, other_symbol]
                        if abs(corr) > correlation_threshold:
                            correlated_group.append(other_symbol)
                            processed.add(other_symbol)

            if len(correlated_group) > 1:
                # Calculate total exposure in correlated group
                group_exposure = sum(positions.get(sym, 0) for sym in correlated_group)

                results['correlated_groups'].append({
                    'symbols': correlated_group,
                    'total_exposure': group_exposure,
                    'average_correlation': correlation_matrix.loc[
                        correlated_group, correlated_group
                    ].mean().mean() if len(correlated_group) > 1 else 0
                })

                results['max_correlated_exposure'] = max(
                    results['max_correlated_exposure'],
                    group_exposure
                )

                # Check violation
                if group_exposure > self.limits.max_correlated_group:
                    results['violations'].append({
                        'type': 'correlation_concentration',
                        'symbols': correlated_group,
                        'current': group_exposure,
                        'limit': self.limits.max_correlated_group,
                        'action': 'reduce_correlated_exposure'
                    })

        return results


class LeverageController:
    """
    Monitor and control portfolio leverage
    """

    def __init__(self, limits: Optional[PortfolioLimits] = None):
        self.limits = limits or PortfolioLimits()

    def calculate_leverage_metrics(self,
                                  long_positions: Dict[str, float],
                                  short_positions: Dict[str, float],
                                  account_value: float,
                                  borrowed_amount: float = 0) -> Dict:
        """
        Calculate leverage metrics

        Returns:
            Dictionary with leverage metrics and margin requirements
        """
        # Calculate position values
        long_value = sum(long_positions.values()) * account_value
        short_value = sum(abs(v) for v in short_positions.values()) * account_value

        # Gross leverage = (Long + |Short|) / Equity
        gross_exposure = long_value + short_value
        equity = account_value - borrowed_amount
        gross_leverage = gross_exposure / equity if equity > 0 else 0

        # Net leverage = (Long - Short) / Equity
        net_exposure = long_value - short_value
        net_leverage = net_exposure / equity if equity > 0 else 0

        # Margin calculations
        margin_required = gross_exposure * self.limits.initial_margin_requirement
        maintenance_margin_required = gross_exposure * self.limits.maintenance_margin
        excess_margin = equity - margin_required

        # Build results
        results = {
            'gross_leverage': gross_leverage,
            'net_leverage': net_leverage,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'equity': equity,
            'margin_required': margin_required,
            'maintenance_margin': maintenance_margin_required,
            'excess_margin': excess_margin,
            'margin_utilization': margin_required / equity if equity > 0 else 0,
            'violations': []
        }

        # Check leverage violations
        if gross_leverage > self.limits.max_gross_leverage:
            results['violations'].append({
                'type': 'gross_leverage_exceeded',
                'current': gross_leverage,
                'limit': self.limits.max_gross_leverage,
                'action': 'reduce_positions',
                'reduction_required': gross_exposure - (self.limits.max_gross_leverage * equity)
            })

        if net_leverage > self.limits.max_net_leverage:
            results['violations'].append({
                'type': 'net_leverage_exceeded',
                'current': net_leverage,
                'limit': self.limits.max_net_leverage,
                'action': 'balance_long_short'
            })

        if excess_margin < 0:
            results['violations'].append({
                'type': 'margin_call',
                'margin_deficit': -excess_margin,
                'action': 'deposit_funds_or_reduce_positions'
            })

        return results


class CashReserveManager:
    """
    Manage cash reserve requirements based on market conditions
    """

    def __init__(self, limits: Optional[PortfolioLimits] = None):
        self.limits = limits or PortfolioLimits()
        self.vix_thresholds = {
            'normal': 20,
            'volatile': 30,
            'crisis': 40
        }

    def determine_market_regime(self,
                               vix: float,
                               market_drawdown: float) -> str:
        """
        Determine current market regime

        Returns: 'normal', 'volatile', or 'crisis'
        """
        # VIX-based regime
        if vix > self.vix_thresholds['crisis']:
            return 'crisis'
        elif vix > self.vix_thresholds['volatile']:
            return 'volatile'

        # Drawdown-based override
        if market_drawdown > 0.20:  # 20% drawdown
            return 'crisis'
        elif market_drawdown > 0.10:  # 10% drawdown
            return 'volatile'

        return 'normal'

    def calculate_required_cash_reserve(self,
                                       portfolio_value: float,
                                       market_regime: str,
                                       upcoming_obligations: float = 0) -> Dict:
        """
        Calculate required cash reserves

        Args:
            portfolio_value: Total portfolio value
            market_regime: Current market regime
            upcoming_obligations: Known future cash needs

        Returns:
            Dictionary with cash reserve requirements
        """
        # Base reserve requirement by regime
        reserve_requirements = {
            'normal': self.limits.min_cash_reserve_normal,
            'volatile': self.limits.min_cash_reserve_volatile,
            'crisis': self.limits.min_cash_reserve_crisis
        }

        base_reserve_pct = reserve_requirements.get(market_regime, 0.05)
        base_reserve_amount = portfolio_value * base_reserve_pct

        # Add buffer for obligations
        total_required = base_reserve_amount + upcoming_obligations

        return {
            'market_regime': market_regime,
            'base_reserve_pct': base_reserve_pct,
            'base_reserve_amount': base_reserve_amount,
            'upcoming_obligations': upcoming_obligations,
            'total_required': total_required,
            'total_required_pct': total_required / portfolio_value if portfolio_value > 0 else 0
        }

    def calculate_cash_deployment_capacity(self,
                                          current_cash: float,
                                          required_reserve: float,
                                          market_regime: str) -> Dict:
        """
        Calculate how much cash can be deployed for new positions
        """
        available_cash = max(0, current_cash - required_reserve)

        # Apply deployment limits based on regime
        deployment_limits = {
            'normal': 0.80,    # Can deploy 80% of available cash
            'volatile': 0.50,  # Can deploy 50% of available cash
            'crisis': 0.25     # Can deploy 25% of available cash
        }

        max_deployment_pct = deployment_limits.get(market_regime, 0.50)
        max_deployment = available_cash * max_deployment_pct

        return {
            'current_cash': current_cash,
            'required_reserve': required_reserve,
            'available_cash': available_cash,
            'max_deployment_pct': max_deployment_pct,
            'max_deployment': max_deployment,
            'reserve_buffer': current_cash - required_reserve
        }