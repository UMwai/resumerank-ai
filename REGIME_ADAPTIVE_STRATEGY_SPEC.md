# REGIME-ADAPTIVE STRATEGY SWITCHING SPECIFICATION
## Comprehensive Framework for um-trading-assistance Platform

---

## Executive Summary

This document specifies a sophisticated regime-adaptive strategy switching system designed to optimize performance across varying market conditions. Based on empirical backtesting results showing passive strategies achieving +33.19% alpha (1.443 Sharpe) in bull markets and active momentum strategies delivering +13% alpha in bear markets (versus -37% for passive), this framework dynamically allocates between strategies based on real-time regime detection.

### Key Performance Targets
- **Bull Market**: Target 1.4+ Sharpe ratio with passive-dominant allocation
- **Bear Market**: Outperform passive by 40%+ through active momentum
- **Overall**: Maintain >75% allocation to prevent SPY underperformance
- **Regime Detection**: <3% false positive rate on transitions

---

## 1. REGIME DETECTION FRAMEWORK

### 1.1 Core Indicators

#### Volatility Regime (35% weight)
```python
vix_indicators = {
    'vix_level': current_vix,                    # Real-time VIX
    'vix_ma20': vix_20day_average,              # Smoothed VIX
    'vix_percentile': percentile_252day,         # Historical context
    'vix_term_structure': vix9d / vix30d,       # Contango/backwardation
    'realized_vol': historical_volatility_20d    # Actual vs implied
}
```

#### Trend Strength (30% weight)
```python
trend_indicators = {
    'sma_positioning': {
        'spy_vs_sma50': spy_close / sma50,
        'spy_vs_sma200': spy_close / sma200,
        'golden_cross': sma50 > sma200
    },
    'trend_slopes': {
        'sma50_slope': linear_regression_slope(sma50, 20),
        'sma200_slope': linear_regression_slope(sma200, 20)
    }
}
```

#### Market Breadth (25% weight)
```python
breadth_indicators = {
    'advance_decline': advancing_stocks / declining_stocks,
    'new_highs_lows': new_52w_highs / new_52w_lows,
    'percent_above_ma': {
        'ma50': stocks_above_50ma / total_stocks,
        'ma200': stocks_above_200ma / total_stocks
    },
    'mcclellan_oscillator': calculate_mcclellan()
}
```

#### Momentum (10% weight)
```python
momentum_indicators = {
    'rsi_14': calculate_rsi(spy_close, 14),
    'macd': calculate_macd_signal(),
    'roc_20': (close / close_20d_ago - 1) * 100
}
```

### 1.2 Regime Classification Thresholds

| Regime | VIX Range | Breadth (>MA50) | Trend | Momentum |
|--------|-----------|-----------------|-------|----------|
| **STRONG BULL** | < 15 | > 70% | SMA50 > SMA200, slope > 0.1%/day | RSI 50-70 |
| **BULL** | 15-20 | 55-70% | Price > SMA50 > SMA200 | RSI 45-65 |
| **SIDEWAYS** | 18-25 | 45-55% | |slope| < 0.05%/day | RSI 40-60 |
| **VOLATILE** | > 25 | Any | High dispersion | Extreme swings |
| **BEAR** | 20-30 | < 45% | SMA50 < SMA200, slope < -0.1%/day | RSI 30-50 |
| **CRASH** | > 30 | < 30% | Accelerating decline | RSI < 30 |

### 1.3 State Transition Logic

```python
class RegimeTransitionLogic:
    def validate_transition(self, current_regime, new_regime, indicators):
        """
        Prevent whipsaw with multi-day confirmation
        """
        # Require 3 consecutive days of new regime signals
        if self.confirmation_counter < 3:
            return current_regime

        # Enforce minimum 5-day stability in current regime
        if self.days_in_regime < 5:
            return current_regime

        # Require 65% confidence threshold
        if self.regime_confidence < 0.65:
            return current_regime

        # Special rules for extreme transitions
        if self._is_extreme_transition(current_regime, new_regime):
            # Require 5-day confirmation for bull->crash or crash->bull
            if self.confirmation_counter < 5:
                return current_regime

        return new_regime
```

### 1.4 Anti-Whipsaw Mechanisms

1. **Hysteresis Bands**: 10% buffer zones between regime thresholds
2. **Smoothing**: Apply 3-day EMA to raw indicators
3. **Ensemble Voting**: Require 3/4 indicator groups to agree
4. **Confidence Decay**: Old regime confidence decays at 10%/day

---

## 2. STRATEGY ALLOCATION BY REGIME

### 2.1 Strong Bull Market Allocation

**Market Characteristics**: Low volatility, strong trend, broad participation

```python
strong_bull_allocation = {
    # Asset Allocation
    'equities': 85%,
    'bonds': 10%,
    'cash': 5%,

    # Strategy Mix (Passive dominates per backtesting)
    'strategies': {
        'passive_low_correlation': 75%,  # Core holding
        'momentum_overlay': 20%,          # Capture trends
        'defensive_hedge': 5%             # Tail protection
    },

    # Implementation Details
    'passive_portfolio': {
        'selection': 'Low correlation stocks with SPY < 0.6',
        'rebalance': 'Quarterly',
        'position_limits': '8% max per position',
        'sector_limits': '30% max per sector'
    },

    # Risk Management
    'risk_params': {
        'stop_loss': None,  # No stops in strong trend
        'trailing_stop': 12%,
        'position_sizing': 'Equal weight * momentum score',
        'max_leverage': 1.0
    }
}
```

**Entry Rules**:
- Buy on any 2% pullback from 10-day high
- Scale in over 3 days for new positions
- Ignore overbought RSI (can stay overbought in strong trends)

### 2.2 Bull Market Allocation

**Market Characteristics**: Moderate volatility, uptrend, good breadth

```python
bull_allocation = {
    # Asset Allocation
    'equities': 75%,
    'bonds': 20%,
    'cash': 5%,

    # Strategy Mix (Balanced approach)
    'strategies': {
        'passive_core': 60%,
        'momentum_satellite': 30%,
        'mean_reversion': 5%,
        'defensive': 5%
    },

    # Risk Management
    'risk_params': {
        'stop_loss': 10%,
        'trailing_stop': 8%,
        'position_sizing': 'Modified Kelly (f/4)',
        'max_leverage': 1.0
    }
}
```

### 2.3 Sideways Market Allocation

**Market Characteristics**: Range-bound, no clear trend

```python
sideways_allocation = {
    # Asset Allocation
    'equities': 50%,
    'bonds': 30%,
    'cash': 15%,
    'alternatives': 5%,  # Gold, commodities

    # Strategy Mix (Mean reversion dominates)
    'strategies': {
        'mean_reversion': 50%,  # Buy oversold, sell overbought
        'passive': 20%,
        'pairs_trading': 15%,
        'momentum': 10%,
        'defensive': 5%
    },

    # Mean Reversion Rules
    'mean_reversion_params': {
        'entry_rsi': [25, 75],  # Buy < 25, Sell > 75
        'bollinger_bands': 2.0,  # Trade at 2-sigma bands
        'holding_period': '5-15 days',
        'position_size': '2-4% per trade'
    }
}
```

### 2.4 Volatile Market Allocation

**Market Characteristics**: High VIX, uncertainty, rapid swings

```python
volatile_allocation = {
    # Asset Allocation (Defensive)
    'equities': 40%,
    'bonds': 35%,
    'cash': 20%,
    'alternatives': 5%,

    # Strategy Mix (Defensive with opportunistic)
    'strategies': {
        'defensive': 50%,
        'short_term_momentum': 20%,
        'mean_reversion': 20%,
        'passive': 10%
    },

    # Risk Management (Tight)
    'risk_params': {
        'stop_loss': 5%,
        'trailing_stop': 4%,
        'position_sizing': '1-3% max',
        'max_leverage': 0.75,
        'rebalance_frequency': 'Weekly'
    }
}
```

### 2.5 Bear Market Allocation

**Market Characteristics**: Downtrend, poor breadth, elevated VIX

```python
bear_allocation = {
    # Asset Allocation (Defensive)
    'equities': 30%,  # Quality defensive stocks only
    'bonds': 45%,     # Treasury flight to quality
    'cash': 20%,
    'alternatives': 5%,

    # Strategy Mix (Active momentum per backtesting)
    'strategies': {
        'active_momentum': 60%,  # +13% alpha in 2022
        'defensive': 30%,
        'mean_reversion': 10%,
        'passive': 0%  # Passive underperforms in bear
    },

    # Active Momentum Implementation
    'momentum_rules': {
        'long_criteria': 'Only top 5% relative strength',
        'short_criteria': 'Bottom 10% relative strength',
        'rebalance': 'Every 10 days',
        'position_sizing': '2-4% per position'
    }
}
```

### 2.6 Crash Allocation

**Market Characteristics**: VIX > 30, panic selling, capitulation

```python
crash_allocation = {
    # Asset Allocation (Maximum defense)
    'equities': 10%,     # Only highest quality
    'bonds': 50%,        # Government bonds
    'cash': 35%,
    'gold': 5%,

    # Strategy Mix
    'strategies': {
        'defensive': 70%,
        'contrarian_buying': 20%,  # Selective on extreme oversold
        'short_momentum': 10%
    },

    # Crisis Management
    'crisis_rules': {
        'new_positions': 'Only on VIX > 40 and RSI < 20',
        'position_size': '1-2% max',
        'stop_loss': 3%,
        'daily_loss_limit': 2%
    }
}
```

---

## 3. TRANSITION MANAGEMENT

### 3.1 Smooth Transition Algorithm

```python
def execute_regime_transition(current_regime, target_regime, portfolio):
    """
    Smoothly transition between regime allocations
    """
    transition_days = calculate_transition_period(current_regime, target_regime)

    # Create daily adjustment path
    daily_adjustments = []
    for day in range(transition_days):
        weight = smooth_weight_function(day, transition_days)

        daily_allocation = interpolate_allocations(
            current_allocation=regime_allocations[current_regime],
            target_allocation=regime_allocations[target_regime],
            weight=weight
        )

        daily_adjustments.append(daily_allocation)

    return daily_adjustments

def calculate_transition_period(from_regime, to_regime):
    """
    Determine optimal transition period
    """
    urgency_matrix = {
        ('bull', 'sideways'): 3,
        ('bull', 'bear'): 2,
        ('bull', 'crash'): 1,  # Urgent
        ('sideways', 'bull'): 3,
        ('sideways', 'bear'): 2,
        ('bear', 'crash'): 1,
        ('crash', 'bull'): 5,  # Gradual recovery
    }

    return urgency_matrix.get((from_regime, to_regime), 3)
```

### 3.2 Transaction Cost Optimization

```python
class TransactionCostOptimizer:
    def optimize_rebalancing(self, current_positions, target_allocation):
        """
        Minimize transaction costs during transitions
        """
        # Calculate position differences
        trades_required = self.calculate_trades(current_positions, target_allocation)

        # Apply optimization rules
        optimized_trades = []

        for trade in trades_required:
            # Skip small trades (< 1% of portfolio)
            if abs(trade.size) < 0.01:
                continue

            # Batch similar trades
            if self.can_batch(trade):
                trade = self.batch_trade(trade)

            # Time trades for liquidity
            trade.execution_time = self.optimal_execution_time(trade)

            # Use limit orders for non-urgent trades
            if not trade.urgent:
                trade.order_type = 'LIMIT'
                trade.limit_price = self.calculate_limit_price(trade)

            optimized_trades.append(trade)

        return optimized_trades

    def estimate_costs(self, trades):
        """
        Estimate total transaction costs
        """
        total_cost = 0

        for trade in trades:
            # Commission
            commission = trade.size * 0.0005  # 5 bps

            # Spread cost
            spread_cost = trade.size * self.get_spread(trade.symbol)

            # Market impact (square root model)
            impact = spread_cost * sqrt(trade.size / adv)

            total_cost += commission + spread_cost + impact

        return total_cost
```

### 3.3 Risk Management During Transitions

```python
class TransitionRiskManager:
    def manage_transition_risk(self, transition_plan):
        """
        Monitor and manage risks during regime transitions
        """
        risk_metrics = {
            'tracking_error': self.calculate_tracking_error(transition_plan),
            'interim_var': self.calculate_transition_var(transition_plan),
            'liquidity_score': self.assess_liquidity_risk(transition_plan),
            'max_interim_exposure': self.calculate_max_exposure(transition_plan)
        }

        # Apply risk limits
        if risk_metrics['tracking_error'] > 0.05:  # 5% tracking error limit
            transition_plan = self.reduce_transition_speed(transition_plan)

        if risk_metrics['liquidity_score'] < 0.7:  # Liquidity concern
            transition_plan = self.split_large_trades(transition_plan)

        # Implement hedges during transition
        if risk_metrics['interim_var'] > self.var_limit:
            hedges = self.calculate_transition_hedges(transition_plan)
            transition_plan.add_hedges(hedges)

        return transition_plan
```

---

## 4. BACKTESTING VALIDATION

### 4.1 Walk-Forward Validation Framework

```python
class WalkForwardValidator:
    def validate_strategy(self, historical_data):
        """
        Implement walk-forward analysis to prevent overfitting
        """
        results = []

        # Define windows
        in_sample_period = 252  # 1 year
        out_sample_period = 63   # 3 months
        step_size = 21           # 1 month

        for start_date in range(0, len(historical_data) - in_sample_period - out_sample_period, step_size):
            # In-sample optimization
            in_sample_data = historical_data[start_date:start_date + in_sample_period]
            optimized_params = self.optimize_parameters(in_sample_data)

            # Out-of-sample testing
            out_sample_data = historical_data[
                start_date + in_sample_period:
                start_date + in_sample_period + out_sample_period
            ]

            performance = self.run_backtest(out_sample_data, optimized_params)
            results.append(performance)

        # Analyze consistency
        return self.analyze_walk_forward_results(results)
```

### 4.2 Performance Metrics

```python
def calculate_comprehensive_metrics(returns, positions, trades):
    """
    Calculate institutional-grade performance metrics
    """
    metrics = {
        # Returns
        'total_return': calculate_total_return(returns),
        'cagr': calculate_cagr(returns),
        'volatility': calculate_volatility(returns),

        # Risk-adjusted
        'sharpe_ratio': calculate_sharpe(returns, risk_free_rate=0.05),
        'sortino_ratio': calculate_sortino(returns, mar=0),
        'calmar_ratio': calculate_calmar(returns),

        # Drawdown
        'max_drawdown': calculate_max_drawdown(returns),
        'average_drawdown': calculate_avg_drawdown(returns),
        'recovery_time': calculate_recovery_time(returns),

        # Regime-specific
        'bull_sharpe': calculate_regime_sharpe(returns, 'bull'),
        'bear_sharpe': calculate_regime_sharpe(returns, 'bear'),
        'regime_hit_rate': calculate_regime_accuracy(predictions, actuals),

        # Trading
        'win_rate': calculate_win_rate(trades),
        'profit_factor': calculate_profit_factor(trades),
        'avg_win_loss_ratio': calculate_win_loss_ratio(trades),

        # Costs
        'total_transaction_costs': sum(trade.cost for trade in trades),
        'cost_per_trade': calculate_avg_cost(trades),
        'implementation_shortfall': calculate_shortfall(trades)
    }

    return metrics
```

### 4.3 Stress Testing Scenarios

```python
stress_scenarios = {
    'covid_crash': {
        'vix_spike': 82,
        'spy_drawdown': -34%,
        'correlation_breakdown': True,
        'liquidity_crisis': True
    },

    'dot_com_bubble': {
        'duration_months': 30,
        'total_drawdown': -49%,
        'growth_underperformance': -75%
    },

    'gfc_2008': {
        'vix_peak': 89,
        'spy_drawdown': -57%,
        'credit_freeze': True,
        'sector_rotation': 'financials_to_utilities'
    },

    'flash_crash': {
        'intraday_drop': -9%,
        'recovery_time': '20 minutes',
        'liquidity_evaporation': True
    },

    'taper_tantrum': {
        'bond_volatility': 'extreme',
        'correlation_flip': 'stocks_bonds_positive'
    }
}

def run_stress_tests(strategy, scenarios):
    """
    Test strategy robustness under extreme scenarios
    """
    results = {}

    for scenario_name, params in scenarios.items():
        scenario_data = generate_scenario_data(params)
        performance = backtest_strategy(strategy, scenario_data)

        results[scenario_name] = {
            'survival': performance['max_drawdown'] > -50%,
            'recovery_time': performance['recovery_days'],
            'sharpe_during': performance['crisis_sharpe'],
            'regime_accuracy': performance['regime_detection_accuracy']
        }

    return results
```

---

## 5. IMPLEMENTATION ROADMAP

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Implement RegimeDetector class with all indicators
2. Create data pipeline for real-time indicator calculation
3. Set up indicator historical database
4. Build backtesting framework

### Phase 2: Strategy Implementation (Weeks 3-4)
1. Code strategy allocation logic for each regime
2. Implement position sizing algorithms
3. Create trade execution engine
4. Build transition management system

### Phase 3: Testing & Validation (Weeks 5-6)
1. Run historical backtests (2020-2024)
2. Perform walk-forward validation
3. Execute stress testing scenarios
4. Optimize parameters

### Phase 4: Production Deployment (Weeks 7-8)
1. Deploy to paper trading environment
2. Monitor real-time performance
3. Implement monitoring dashboard
4. Set up alerting system

---

## 6. MONITORING & MAINTENANCE

### 6.1 Real-Time Monitoring Dashboard

```python
monitoring_metrics = {
    'regime_status': {
        'current_regime': 'BULL',
        'confidence': 0.78,
        'days_in_regime': 23,
        'transition_probability': 0.15
    },

    'performance_tracking': {
        'daily_pnl': calculate_daily_pnl(),
        'mtd_return': calculate_mtd_return(),
        'rolling_sharpe': calculate_rolling_sharpe(window=30),
        'vs_benchmark': calculate_vs_spy()
    },

    'risk_metrics': {
        'current_var': calculate_var_95(),
        'portfolio_beta': calculate_portfolio_beta(),
        'concentration_risk': calculate_hhi(),
        'correlation_matrix': calculate_correlations()
    },

    'execution_quality': {
        'slippage': calculate_slippage(),
        'fill_rate': calculate_fill_rate(),
        'transaction_costs': calculate_costs()
    }
}
```

### 6.2 Alert Conditions

```python
alert_rules = {
    'regime_change_imminent': lambda: confidence < 0.6 and confirmation_days >= 2,
    'drawdown_threshold': lambda: current_drawdown > max_drawdown_limit * 0.8,
    'correlation_breakdown': lambda: avg_correlation > 0.8,
    'liquidity_warning': lambda: bid_ask_spread > normal_spread * 2,
    'var_breach': lambda: current_var > var_limit,
    'position_concentration': lambda: max_position > position_limit * 0.9
}
```

---

## 7. APPENDIX: CONFIGURATION TEMPLATES

### 7.1 Development Configuration

```yaml
regime_detection:
  confirmation_days: 3
  min_confidence: 0.65
  smoothing_window: 3

portfolio:
  initial_capital: 100000
  max_positions: 30
  rebalance_frequency: weekly

risk_limits:
  max_drawdown: 0.20
  position_limit: 0.08
  sector_limit: 0.30
  var_limit: 0.02

execution:
  slippage_model: linear
  commission: 0.0005
  market_impact: sqrt
```

### 7.2 Production Configuration

```yaml
regime_detection:
  confirmation_days: 5
  min_confidence: 0.70
  smoothing_window: 5

portfolio:
  initial_capital: 1000000
  max_positions: 50
  rebalance_frequency: daily

risk_limits:
  max_drawdown: 0.15
  position_limit: 0.05
  sector_limit: 0.25
  var_limit: 0.015

execution:
  slippage_model: non_linear
  commission: 0.0003
  market_impact: almgren_chriss
  urgency_premium: true
```

---

## CONCLUSION

This regime-adaptive strategy framework provides a robust, institutional-grade approach to systematic trading that adapts to changing market conditions. By combining sophisticated regime detection with optimized strategy allocation and careful transition management, the system aims to deliver consistent risk-adjusted returns across all market environments.

Key success factors:
1. **Accurate regime detection** with low false positive rate
2. **Smooth transitions** that minimize costs and risks
3. **Appropriate strategy selection** based on empirical evidence
4. **Robust risk management** throughout all phases
5. **Continuous monitoring and adaptation**

The framework is designed to be implemented incrementally, tested thoroughly, and deployed with appropriate safeguards. Regular monitoring and periodic re-optimization ensure continued effectiveness as market dynamics evolve.