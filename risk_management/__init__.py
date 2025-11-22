"""
Advanced Risk Management Framework for Trading Assistance Platform
Author: Quantitative Risk Management Expert
Version: 1.0.0
"""

from .position_sizing import (
    KellyCriterion,
    VolatilityAdjustedSizing,
    CorrelationAdjustedSizing,
    PositionSizeCalculator
)

from .portfolio_controls import (
    VaRMonitor,
    CVaRCalculator,
    ConcentrationLimits,
    LeverageController,
    CashReserveManager
)

from .circuit_breakers import (
    DrawdownCircuitBreaker,
    LossCircuitBreaker,
    VolatilityCircuitBreaker,
    CircuitBreakerManager
)

from .stress_testing import (
    StressTester,
    ScenarioGenerator,
    RecoveryAnalyzer,
    MonteCarloSimulator
)

from .dynamic_adjustment import (
    VolatilityRegimeDetector,
    DynamicRiskAdjuster,
    MarketConditionAnalyzer,
    AdaptiveStopLossManager
)

__all__ = [
    'KellyCriterion',
    'VolatilityAdjustedSizing',
    'CorrelationAdjustedSizing',
    'PositionSizeCalculator',
    'VaRMonitor',
    'CVaRCalculator',
    'ConcentrationLimits',
    'LeverageController',
    'CashReserveManager',
    'DrawdownCircuitBreaker',
    'LossCircuitBreaker',
    'VolatilityCircuitBreaker',
    'CircuitBreakerManager',
    'StressTester',
    'ScenarioGenerator',
    'RecoveryAnalyzer',
    'MonteCarloSimulator',
    'VolatilityRegimeDetector',
    'DynamicRiskAdjuster',
    'MarketConditionAnalyzer',
    'AdaptiveStopLossManager'
]