# Advanced Risk Management Framework

## Overview

This is a comprehensive, institutional-grade risk management framework designed for algorithmic trading platforms managing portfolios from $50K to $500K+. The framework provides sophisticated risk controls, dynamic adjustments, and stress testing capabilities to maintain strict risk parameters.

## Target Performance Metrics

- **Maximum Drawdown**: < 15%
- **Sharpe Ratio**: > 1.5
- **Recovery Time**: < 6 months from 15% drawdown
- **Portfolio Survival Rate**: > 95% in stress scenarios

## Framework Components

### 1. Position Sizing Framework (`position_sizing.py`)

Advanced position sizing algorithms including:

#### Kelly Criterion Implementation
- **Unconstrained Kelly**: Full Kelly calculation for optimal growth
- **Constrained Kelly**: Limited to 25% (1/4 Kelly) for safety
- **Multi-Asset Kelly**: Portfolio-wide optimization
- **Confidence Adjustment**: Reduces sizing based on parameter uncertainty

#### Volatility-Adjusted Sizing
- **Target Volatility**: 15% annualized
- **Dynamic Scaling**: Position sizes inversely proportional to volatility
- **ATR-Based Sizing**: Uses Average True Range for intraday adjustments
- **EWM Volatility**: Exponentially weighted for responsiveness

#### Correlation-Adjusted Sizing
- **Correlation Threshold**: 0.7 for high correlation detection
- **Cluster Analysis**: Groups correlated assets
- **Maximum Correlated Exposure**: 25% in highly correlated positions
- **Automatic Scaling**: Reduces positions in correlation clusters

#### Position Limits by Strategy Type
```python
# Strategy-specific maximum positions
momentum_max = 10%          # Trend following
mean_reversion_max = 8%     # Counter-trend
event_driven_max = 12%      # News/earnings based
arbitrage_max = 15%         # Pairs/statistical arb

# Risk tier limits (by volatility percentile)
low_risk_max = 15%         # 0-33rd percentile
medium_risk_max = 10%      # 33-66th percentile
high_risk_max = 5%         # 66-100th percentile
```

### 2. Portfolio-Level Risk Controls (`portfolio_controls.py`)

Real-time portfolio monitoring and risk limits:

#### VaR Monitoring (Value at Risk)
- **95% Confidence Level**: Maximum 5% daily VaR
- **99% Confidence Level**: Maximum 8% daily VaR
- **Methods**: Historical, Parametric, and Monte Carlo VaR
- **Combined Approach**: Uses maximum of all methods (conservative)

#### CVaR Calculation (Conditional VaR)
- **95% Confidence Level**: Maximum 7% daily CVaR
- **99% Confidence Level**: Maximum 10% daily CVaR
- **Expected Shortfall**: Average loss beyond VaR threshold
- **CVaR/VaR Ratio**: Must be > 1.0 (tail risk check)

#### Concentration Limits
```python
# Maximum exposures
max_sector = 35%           # Any single sector
max_stock = 15%            # Any single position
max_correlated = 40%       # Correlated asset groups (>0.6 correlation)

# Herfindahl Index monitoring
# Effective N = 1/HI (target > 10 for diversification)
```

#### Leverage Controls
- **Maximum Gross Leverage**: 1.5x (150% gross exposure)
- **Maximum Net Leverage**: 1.0x (100% net exposure)
- **Initial Margin**: 50% requirement
- **Maintenance Margin**: 25% requirement
- **Automatic Deleveraging**: Triggered at margin breach

#### Cash Reserve Management
```python
# Market regime-based cash requirements
Normal Markets:    5% minimum cash
Volatile Markets: 15% minimum cash (VIX 20-30)
Crisis Markets:   30% minimum cash (VIX > 40)

# Deployment limits
Normal:    Can deploy 80% of excess cash
Volatile:  Can deploy 50% of excess cash
Crisis:    Can deploy 25% of excess cash
```

### 3. Circuit Breakers & Kill Switches (`circuit_breakers.py`)

Automatic trading halts with escalating severity:

#### Drawdown Circuit Breakers
```python
# Portfolio drawdown triggers
Warning:  8% drawdown  → Reduce position sizes
Halt:    12% drawdown  → Stop all new positions
Kill:    15% drawdown  → Emergency liquidation

# Automatic cooldown periods
Initial halt: 15 minutes
Repeated halt: 30 minutes (escalating)
Kill switch: Manual intervention required
```

#### Daily Loss Circuit Breakers
```python
# Daily P&L triggers
Warning:  3% daily loss → Tighten stop losses
Halt:     5% daily loss → Halt new positions
Kill:     8% daily loss → Close all positions

# Maximum triggers per day: 3
# After 3 triggers → Day halt (no trading until next day)
```

#### Position Loss Circuit Breakers
```python
# Individual position triggers
Warning: 10% position loss → Tighten stop loss
Halt:    15% position loss → Close position
Kill:    20% position loss → Emergency close

# Applied per position independently
```

#### Volatility Circuit Breakers
```python
# VIX-based triggers
Warning: VIX > 30 → Reduce leverage
Halt:    VIX > 40 → Halt risky strategies
Kill:    VIX > 50 → Maximum defensive posture

# Correlation breakdown detection
Threshold: 0.5 correlation change → Halt trading
```

### 4. Stress Testing Framework (`stress_testing.py`)

Comprehensive scenario testing against historical and hypothetical events:

#### Historical Scenarios
1. **2008 Financial Crisis**
   - Market Shock: -50%
   - Duration: 365 days
   - Recovery: 730 days
   - Sector Impact: Financials -70%, Tech -45%

2. **COVID-19 Crash**
   - Market Shock: -35%
   - Duration: 30 days
   - Recovery: 180 days
   - Sector Impact: Travel -70%, Energy -65%

3. **Flash Crash**
   - Market Shock: -10%
   - Duration: 1 day
   - Recovery: 5 days
   - High-frequency trading induced

4. **Dot-Com Bubble**
   - Market Shock: -45%
   - Duration: 900 days
   - Sector Impact: Tech -75%

5. **Black Monday 1987**
   - Market Shock: -22%
   - Duration: 1 day
   - Universal impact

#### Hypothetical Scenarios
- Sovereign Debt Crisis (-40%)
- Cyber Attack on Financial System (-25%)
- Geopolitical Crisis (-30%)
- Inflation Shock (-20%)
- Liquidity Crisis (-15%)

#### Monte Carlo Simulation
- 10,000 simulations per test
- Multi-variate normal returns
- Tail risk modeling (Extreme Value Theory)
- Recovery path analysis

#### Portfolio Requirements
```python
# Must survive all scenarios with:
Maximum Loss: < 50% in worst case
Survival Rate: > 95% of scenarios
Recovery Time: < 2 years for 90% probability
VaR Breach: < 10% frequency
```

### 5. Dynamic Risk Adjustment (`dynamic_adjustment.py`)

Automatic parameter adjustments based on market conditions:

#### Market Regime Detection
```python
# Six regime classifications
BULL_QUIET:     Low volatility uptrend
BULL_VOLATILE:  High volatility uptrend
BEAR_QUIET:     Low volatility downtrend
BEAR_VOLATILE:  High volatility downtrend
RANGING:        Sideways market
CRISIS:         Extreme conditions
```

#### Position Size Adjustments
```python
# Regime-based multipliers
Bull Quiet:    1.2x normal size
Bull Volatile: 0.9x normal size
Bear Quiet:    0.7x normal size
Bear Volatile: 0.5x normal size
Ranging:       0.8x normal size
Crisis:        0.3x normal size
```

#### Stop Loss Adjustments
```python
# Volatility-based multipliers
Bull Quiet:    1.5x normal distance (wider)
Bull Volatile: 1.0x normal distance
Bear Quiet:    0.8x normal distance (tighter)
Bear Volatile: 0.6x normal distance
Crisis:        0.5x normal distance (very tight)
```

#### VIX-Based Triggers
```python
VIX < 15:  Low volatility (increase risk)
VIX 15-25: Normal volatility
VIX 25-35: High volatility (reduce risk)
VIX 35-45: Very high (defensive mode)
VIX > 45:  Crisis mode (minimum risk)
```

#### Adaptive Stop Loss Management
- ATR-based dynamic stops
- Time decay adjustment (tighter over time)
- Breakeven stops for profitable positions
- Trailing stops with volatility adjustment

## Implementation Example

```python
from risk_management import (
    PositionSizeCalculator,
    VaRMonitor,
    CircuitBreakerManager,
    StressTester,
    DynamicRiskAdjuster
)

# Initialize components
position_calculator = PositionSizeCalculator(
    account_value=100000,
    max_portfolio_risk=0.06
)

var_monitor = VaRMonitor(confidence_levels=[0.95, 0.99])
circuit_breakers = CircuitBreakerManager()
stress_tester = StressTester()
risk_adjuster = DynamicRiskAdjuster()

# Calculate position size
risk_metrics = {
    'win_probability': 0.55,
    'avg_win_return': 0.05,
    'avg_loss_return': -0.03,
    'volatility_percentile': 45,
    'historical_returns': returns_series
}

optimal_size = position_calculator.calculate_optimal_position_size(
    strategy_type='momentum',
    expected_return=0.08,
    risk_metrics=risk_metrics,
    current_positions=current_portfolio,
    correlation_matrix=corr_matrix
)

# Monitor portfolio risk
var_results = var_monitor.calculate_combined_var(
    returns=portfolio_returns,
    portfolio_value=100000,
    confidence_level=0.95
)

# Check circuit breakers
breaker_status = circuit_breakers.update_all_metrics(
    portfolio_value=95000,
    starting_value=100000,
    positions=current_positions,
    vix=28
)

if not breaker_status['can_trade']:
    print(f"Trading halted: {breaker_status['action']}")

# Run stress tests
stress_results = stress_tester.run_all_scenarios(
    portfolio_positions=current_positions,
    position_betas=betas,
    sector_mapping=sectors
)

print(f"Portfolio survival rate: {stress_results['summary']['survival_rate']:.1%}")

# Get dynamic adjustments
adjustments = risk_adjuster.calculate_risk_adjustments(
    market_data=market_df,
    vix=28,
    current_positions=current_positions,
    account_value=100000
)

print(f"Current regime: {adjustments['current_regime']}")
print(f"Position size multiplier: {adjustments['position_size_adjustment']['weighted_multiplier']:.2f}")
```

## Risk Management Workflow

### Daily Pre-Market Routine
1. Check overnight market movements
2. Update VIX and volatility metrics
3. Run regime detection
4. Adjust position size limits
5. Review circuit breaker status

### Intraday Monitoring
1. Real-time VaR tracking
2. Position P&L monitoring
3. Circuit breaker checks (every 5 minutes)
4. Correlation monitoring
5. Margin utilization tracking

### End-of-Day Analysis
1. Calculate daily returns
2. Update drawdown metrics
3. Run stress tests
4. Generate risk report
5. Plan next day adjustments

### Weekly Review
1. Full portfolio stress testing
2. Regime probability analysis
3. Risk metric trends
4. Circuit breaker event review
5. Position correlation analysis

## Performance Metrics

### Risk-Adjusted Returns
- **Sharpe Ratio Target**: > 1.5
- **Sortino Ratio Target**: > 2.0
- **Calmar Ratio Target**: > 1.0
- **Information Ratio Target**: > 0.5

### Drawdown Management
- **Maximum Drawdown**: < 15%
- **Average Drawdown**: < 8%
- **Drawdown Duration**: < 60 days
- **Recovery Rate**: > 2x drawdown speed

### Risk Metrics
- **Daily VaR (95%)**: < 5%
- **Monthly VaR (95%)**: < 10%
- **Tail Risk (CVaR/VaR)**: < 1.5
- **Beta to Market**: < 0.8

## Testing

Run the test suite:

```bash
pytest tests/
pytest tests/test_position_sizing.py -v
pytest tests/test_circuit_breakers.py -v
pytest --cov=risk_management tests/
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Create a `config.yaml` file:

```yaml
risk_management:
  kelly_fraction: 0.25
  target_volatility: 0.15
  max_drawdown: 0.15
  var_confidence: 0.95

circuit_breakers:
  portfolio_drawdown_halt: 0.12
  daily_loss_halt: 0.05
  position_loss_halt: 0.15

market_regimes:
  vix_crisis_threshold: 40
  vix_high_threshold: 30
  vix_normal_threshold: 20
```

## Best Practices

1. **Never Override Circuit Breakers** without senior approval
2. **Always Run Stress Tests** before deploying new strategies
3. **Monitor Correlation Changes** daily
4. **Maintain Minimum Cash Reserves** per regime
5. **Document All Manual Overrides** with reasoning
6. **Review Risk Metrics** at least twice daily
7. **Test New Strategies** with reduced sizing first

## Support

For questions or issues, please refer to the documentation or create an issue in the repository.

## License

Proprietary - Internal Use Only