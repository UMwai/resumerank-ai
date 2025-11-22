"""
Stress Testing Framework
Test portfolio resilience under extreme market scenarios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import norm, t, gumbel_r
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Definition of a stress test scenario"""
    name: str
    description: str
    market_shock: float          # Market decline percentage
    volatility_multiplier: float # VIX multiplier
    correlation_shift: float     # Increase in correlations
    duration_days: int          # Scenario duration
    recovery_days: int          # Recovery period
    sector_impacts: Dict[str, float]  # Sector-specific impacts
    probability: float          # Estimated probability (for risk weighting)


class ScenarioGenerator:
    """
    Generate stress test scenarios based on historical events and hypothetical situations
    """

    def __init__(self):
        self.historical_scenarios = self._load_historical_scenarios()
        self.hypothetical_scenarios = self._generate_hypothetical_scenarios()

    def _load_historical_scenarios(self) -> List[StressScenario]:
        """Load historical crisis scenarios"""
        scenarios = [
            StressScenario(
                name="2008 Financial Crisis",
                description="Global financial system collapse, Lehman bankruptcy",
                market_shock=-0.50,
                volatility_multiplier=4.0,
                correlation_shift=0.3,
                duration_days=365,
                recovery_days=730,
                sector_impacts={
                    'Financials': -0.70,
                    'Real Estate': -0.60,
                    'Energy': -0.55,
                    'Consumer Discretionary': -0.50,
                    'Technology': -0.45,
                    'Healthcare': -0.35,
                    'Consumer Staples': -0.25,
                    'Utilities': -0.30
                },
                probability=0.02
            ),
            StressScenario(
                name="COVID-19 Crash",
                description="Pandemic-induced market crash and rapid recovery",
                market_shock=-0.35,
                volatility_multiplier=3.5,
                correlation_shift=0.4,
                duration_days=30,
                recovery_days=180,
                sector_impacts={
                    'Travel & Leisure': -0.70,
                    'Energy': -0.65,
                    'Financials': -0.45,
                    'Real Estate': -0.40,
                    'Consumer Discretionary': -0.35,
                    'Technology': -0.25,
                    'Healthcare': -0.20,
                    'Consumer Staples': -0.15
                },
                probability=0.01
            ),
            StressScenario(
                name="Flash Crash",
                description="High-frequency trading induced flash crash",
                market_shock=-0.10,
                volatility_multiplier=2.5,
                correlation_shift=0.5,
                duration_days=1,
                recovery_days=5,
                sector_impacts={
                    'Technology': -0.15,
                    'Financials': -0.12,
                    'All': -0.10
                },
                probability=0.05
            ),
            StressScenario(
                name="Dot-Com Bubble Burst",
                description="Technology bubble collapse 2000-2002",
                market_shock=-0.45,
                volatility_multiplier=2.5,
                correlation_shift=0.2,
                duration_days=900,
                recovery_days=1800,
                sector_impacts={
                    'Technology': -0.75,
                    'Telecommunications': -0.60,
                    'Financials': -0.35,
                    'Consumer Discretionary': -0.30,
                    'Energy': -0.20,
                    'Healthcare': -0.25,
                    'Consumer Staples': -0.15
                },
                probability=0.03
            ),
            StressScenario(
                name="Black Monday 1987",
                description="Single day 22% market crash",
                market_shock=-0.22,
                volatility_multiplier=5.0,
                correlation_shift=0.6,
                duration_days=1,
                recovery_days=365,
                sector_impacts={
                    'All': -0.22
                },
                probability=0.01
            )
        ]
        return scenarios

    def _generate_hypothetical_scenarios(self) -> List[StressScenario]:
        """Generate hypothetical stress scenarios"""
        scenarios = [
            StressScenario(
                name="Sovereign Debt Crisis",
                description="Major developed country debt default",
                market_shock=-0.40,
                volatility_multiplier=3.0,
                correlation_shift=0.35,
                duration_days=180,
                recovery_days=730,
                sector_impacts={
                    'Financials': -0.60,
                    'Government Contractors': -0.50,
                    'Utilities': -0.35,
                    'All': -0.40
                },
                probability=0.02
            ),
            StressScenario(
                name="Cyber Attack on Financial System",
                description="Major cyber attack disrupting financial infrastructure",
                market_shock=-0.25,
                volatility_multiplier=3.5,
                correlation_shift=0.4,
                duration_days=30,
                recovery_days=90,
                sector_impacts={
                    'Financials': -0.45,
                    'Technology': -0.35,
                    'Payments': -0.50,
                    'All': -0.25
                },
                probability=0.03
            ),
            StressScenario(
                name="Geopolitical Crisis",
                description="Major military conflict or geopolitical event",
                market_shock=-0.30,
                volatility_multiplier=2.5,
                correlation_shift=0.3,
                duration_days=90,
                recovery_days=365,
                sector_impacts={
                    'Energy': -0.20,  # Could be positive for energy
                    'Defense': 0.10,   # Defense stocks might benefit
                    'Financials': -0.35,
                    'Consumer Discretionary': -0.40,
                    'Travel': -0.50,
                    'All': -0.30
                },
                probability=0.04
            ),
            StressScenario(
                name="Inflation Shock",
                description="Sudden spike in inflation requiring aggressive rate hikes",
                market_shock=-0.20,
                volatility_multiplier=2.0,
                correlation_shift=0.15,
                duration_days=365,
                recovery_days=730,
                sector_impacts={
                    'Technology': -0.35,  # Growth stocks hit hardest
                    'Real Estate': -0.30,
                    'Utilities': -0.25,
                    'Financials': -0.10,  # Banks might benefit
                    'Energy': 0.05,      # Commodities hedge
                    'Materials': 0.05,
                    'Consumer Staples': -0.15
                },
                probability=0.10
            ),
            StressScenario(
                name="Liquidity Crisis",
                description="Severe liquidity crunch in credit markets",
                market_shock=-0.15,
                volatility_multiplier=2.5,
                correlation_shift=0.5,
                duration_days=30,
                recovery_days=180,
                sector_impacts={
                    'Small Caps': -0.35,
                    'High Yield': -0.40,
                    'Emerging Markets': -0.45,
                    'Financials': -0.25,
                    'All': -0.15
                },
                probability=0.05
            )
        ]
        return scenarios

    def get_all_scenarios(self) -> List[StressScenario]:
        """Get all available scenarios"""
        return self.historical_scenarios + self.hypothetical_scenarios

    def get_scenario_by_name(self, name: str) -> Optional[StressScenario]:
        """Get specific scenario by name"""
        all_scenarios = self.get_all_scenarios()
        for scenario in all_scenarios:
            if scenario.name == name:
                return scenario
        return None


class StressTester:
    """
    Main stress testing engine
    """

    def __init__(self):
        self.scenario_generator = ScenarioGenerator()
        self.monte_carlo_sims = 10000

    def run_single_scenario(self,
                           portfolio_positions: Dict[str, float],
                           scenario: StressScenario,
                           position_betas: Dict[str, float],
                           sector_mapping: Dict[str, str]) -> Dict:
        """
        Run a single stress test scenario

        Args:
            portfolio_positions: Dict of symbol -> position value
            scenario: Stress scenario to test
            position_betas: Dict of symbol -> beta to market
            sector_mapping: Dict of symbol -> sector

        Returns:
            Dict with scenario results
        """
        results = {
            'scenario_name': scenario.name,
            'initial_value': sum(portfolio_positions.values()),
            'position_impacts': {},
            'total_loss': 0,
            'total_loss_pct': 0,
            'worst_positions': [],
            'survival': True
        }

        # Calculate impact on each position
        for symbol, position_value in portfolio_positions.items():
            # Get position beta (default to 1.0)
            beta = position_betas.get(symbol, 1.0)

            # Get sector-specific impact
            sector = sector_mapping.get(symbol, 'All')
            sector_impact = scenario.sector_impacts.get(sector,
                                                       scenario.sector_impacts.get('All', scenario.market_shock))

            # Calculate position loss
            # Loss = Beta * Market Shock + Sector Specific Impact
            position_shock = (beta * scenario.market_shock * 0.5) + (sector_impact * 0.5)

            # Add idiosyncratic risk (company-specific)
            idiosyncratic = np.random.normal(0, 0.05)  # 5% idiosyncratic volatility
            position_shock += idiosyncratic

            # Apply correlation shift (increases systematic risk)
            if scenario.correlation_shift > 0:
                position_shock *= (1 + scenario.correlation_shift)

            # Calculate loss
            position_loss = position_value * position_shock
            results['position_impacts'][symbol] = {
                'initial_value': position_value,
                'shock': position_shock,
                'loss': position_loss,
                'final_value': position_value + position_loss
            }

            results['total_loss'] += position_loss

        # Calculate summary statistics
        results['final_value'] = results['initial_value'] + results['total_loss']
        results['total_loss_pct'] = results['total_loss'] / results['initial_value'] if results['initial_value'] > 0 else 0

        # Identify worst hit positions
        position_losses = [(sym, data['loss'], data['shock'])
                          for sym, data in results['position_impacts'].items()]
        position_losses.sort(key=lambda x: x[1])  # Sort by loss (most negative first)

        results['worst_positions'] = [
            {
                'symbol': sym,
                'loss': loss,
                'loss_pct': shock
            }
            for sym, loss, shock in position_losses[:5]
        ]

        # Check survival (portfolio doesn't go to zero or below threshold)
        results['survival'] = results['final_value'] > (results['initial_value'] * 0.50)  # Survive if >50% remains

        # Add volatility and drawdown metrics
        results['max_drawdown'] = min(results['total_loss_pct'], -0.50)  # Cap at -50%
        results['expected_recovery_days'] = scenario.recovery_days
        results['var_breach'] = results['total_loss_pct'] < -0.10  # VaR breach if loss > 10%

        return results

    def run_all_scenarios(self,
                         portfolio_positions: Dict[str, float],
                         position_betas: Dict[str, float],
                         sector_mapping: Dict[str, str]) -> Dict:
        """
        Run all available stress test scenarios

        Returns:
            Dict with results from all scenarios
        """
        all_scenarios = self.scenario_generator.get_all_scenarios()
        results = {
            'portfolio_value': sum(portfolio_positions.values()),
            'scenarios': {},
            'summary': {}
        }

        worst_loss = 0
        worst_scenario = None
        scenarios_failed = 0

        for scenario in all_scenarios:
            scenario_result = self.run_single_scenario(
                portfolio_positions,
                scenario,
                position_betas,
                sector_mapping
            )

            results['scenarios'][scenario.name] = scenario_result

            # Track worst case
            if scenario_result['total_loss_pct'] < worst_loss:
                worst_loss = scenario_result['total_loss_pct']
                worst_scenario = scenario.name

            # Count failures
            if not scenario_result['survival']:
                scenarios_failed += 1

        # Calculate summary statistics
        results['summary'] = {
            'worst_scenario': worst_scenario,
            'worst_loss_pct': worst_loss,
            'scenarios_tested': len(all_scenarios),
            'scenarios_failed': scenarios_failed,
            'survival_rate': (len(all_scenarios) - scenarios_failed) / len(all_scenarios),
            'expected_max_loss': self._calculate_expected_loss(results['scenarios']),
            'risk_score': self._calculate_risk_score(results['scenarios'])
        }

        return results

    def _calculate_expected_loss(self, scenario_results: Dict) -> float:
        """Calculate probability-weighted expected loss"""
        expected_loss = 0
        total_probability = 0

        for scenario_name, result in scenario_results.items():
            # Get scenario probability
            scenario = self.scenario_generator.get_scenario_by_name(scenario_name)
            if scenario:
                expected_loss += result['total_loss_pct'] * scenario.probability
                total_probability += scenario.probability

        if total_probability > 0:
            return expected_loss / total_probability
        return 0

    def _calculate_risk_score(self, scenario_results: Dict) -> float:
        """
        Calculate overall risk score (0-100)
        Higher score = higher risk
        """
        factors = []

        # Factor 1: Average loss across scenarios
        avg_loss = np.mean([r['total_loss_pct'] for r in scenario_results.values()])
        factors.append(min(abs(avg_loss) * 2, 1.0))  # Scale to 0-1

        # Factor 2: Worst case loss
        worst_loss = min([r['total_loss_pct'] for r in scenario_results.values()])
        factors.append(min(abs(worst_loss), 1.0))

        # Factor 3: Survival rate
        survival_rate = sum(1 for r in scenario_results.values() if r['survival']) / len(scenario_results)
        factors.append(1.0 - survival_rate)

        # Factor 4: VaR breach frequency
        var_breaches = sum(1 for r in scenario_results.values() if r['var_breach']) / len(scenario_results)
        factors.append(var_breaches)

        # Weight factors and scale to 0-100
        risk_score = np.average(factors, weights=[0.2, 0.4, 0.3, 0.1]) * 100

        return min(risk_score, 100)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for portfolio risk assessment
    """

    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations

    def simulate_portfolio_returns(self,
                                  expected_returns: np.ndarray,
                                  covariance_matrix: np.ndarray,
                                  time_horizon: int = 252,
                                  initial_value: float = 100000) -> Dict:
        """
        Simulate portfolio returns using Monte Carlo

        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            time_horizon: Number of days to simulate
            initial_value: Initial portfolio value

        Returns:
            Dict with simulation results
        """
        n_assets = len(expected_returns)

        # Generate random returns
        random_returns = np.random.multivariate_normal(
            expected_returns,
            covariance_matrix,
            size=(self.n_simulations, time_horizon)
        )

        # Calculate cumulative portfolio values
        portfolio_values = np.zeros((self.n_simulations, time_horizon + 1))
        portfolio_values[:, 0] = initial_value

        for t in range(time_horizon):
            portfolio_values[:, t + 1] = portfolio_values[:, t] * (1 + random_returns[:, :, t].sum(axis=1))

        # Calculate statistics
        final_values = portfolio_values[:, -1]
        max_values = portfolio_values.max(axis=1)
        min_values = portfolio_values.min(axis=1)

        # Calculate drawdowns
        drawdowns = []
        for sim in range(self.n_simulations):
            running_max = portfolio_values[sim, 0]
            max_drawdown = 0
            for value in portfolio_values[sim, 1:]:
                running_max = max(running_max, value)
                drawdown = (running_max - value) / running_max if running_max > 0 else 0
                max_drawdown = max(max_drawdown, drawdown)
            drawdowns.append(max_drawdown)

        results = {
            'initial_value': initial_value,
            'expected_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'percentiles': {
                '5th': np.percentile(final_values, 5),
                '10th': np.percentile(final_values, 10),
                '25th': np.percentile(final_values, 25),
                '75th': np.percentile(final_values, 75),
                '90th': np.percentile(final_values, 90),
                '95th': np.percentile(final_values, 95)
            },
            'probability_of_loss': np.sum(final_values < initial_value) / self.n_simulations,
            'probability_of_50pct_loss': np.sum(final_values < initial_value * 0.5) / self.n_simulations,
            'max_drawdown': {
                'mean': np.mean(drawdowns),
                'median': np.median(drawdowns),
                '95th_percentile': np.percentile(drawdowns, 95),
                'worst': np.max(drawdowns)
            },
            'var_95': initial_value - np.percentile(final_values, 5),
            'cvar_95': initial_value - np.mean(final_values[final_values <= np.percentile(final_values, 5)]),
            'best_case': np.max(final_values),
            'worst_case': np.min(final_values),
            'sharpe_estimate': self._calculate_sharpe(portfolio_values)
        }

        return results

    def _calculate_sharpe(self, portfolio_values: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from simulated returns"""
        # Calculate daily returns for each simulation
        returns = np.diff(portfolio_values, axis=1) / portfolio_values[:, :-1]

        # Calculate annualized Sharpe for each simulation
        sharpe_ratios = []
        for sim_returns in returns:
            mean_return = np.mean(sim_returns) * 252
            std_return = np.std(sim_returns) * np.sqrt(252)
            if std_return > 0:
                sharpe = (mean_return - risk_free_rate) / std_return
                sharpe_ratios.append(sharpe)

        return np.mean(sharpe_ratios) if sharpe_ratios else 0

    def simulate_tail_risk(self,
                          returns_data: pd.Series,
                          confidence_levels: List[float] = [0.95, 0.99],
                          block_size: int = 10) -> Dict:
        """
        Simulate extreme tail risk using Extreme Value Theory

        Args:
            returns_data: Historical returns
            confidence_levels: Confidence levels for tail risk
            block_size: Block size for block maxima method
        """
        results = {}

        # Fit Generalized Extreme Value distribution to worst returns
        worst_returns = returns_data.nsmallest(int(len(returns_data) * 0.1))

        # Fit GEV distribution
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = gumbel_r.fit(-worst_returns)  # Negative for losses

        for conf_level in confidence_levels:
            # Calculate extreme VaR
            extreme_var = -gumbel_r.ppf(1 - conf_level, *params)
            results[f'extreme_var_{int(conf_level*100)}'] = extreme_var

            # Calculate expected shortfall
            tail_sample = -gumbel_r.rvs(*params, size=1000)
            threshold = np.percentile(tail_sample, (1 - conf_level) * 100)
            expected_shortfall = np.mean(tail_sample[tail_sample <= threshold])
            results[f'extreme_cvar_{int(conf_level*100)}'] = expected_shortfall

        return results


class RecoveryAnalyzer:
    """
    Analyze portfolio recovery characteristics after drawdowns
    """

    def __init__(self):
        self.recovery_scenarios = []

    def analyze_recovery_time(self,
                             drawdown_pct: float,
                             expected_return: float,
                             volatility: float) -> Dict:
        """
        Estimate recovery time from drawdown

        Uses formula: Recovery Return Needed = drawdown / (1 - drawdown)
        """
        # Calculate required return to recover
        recovery_return_needed = abs(drawdown_pct) / (1 + drawdown_pct)

        # Estimate recovery time under different scenarios
        # Optimistic: expected return + 1 std dev
        optimistic_return = expected_return + volatility
        optimistic_time = recovery_return_needed / optimistic_return if optimistic_return > 0 else float('inf')

        # Base case: expected return
        base_time = recovery_return_needed / expected_return if expected_return > 0 else float('inf')

        # Pessimistic: expected return - 1 std dev
        pessimistic_return = max(expected_return - volatility, 0.01)
        pessimistic_time = recovery_return_needed / pessimistic_return

        # Monte Carlo simulation for probabilistic recovery time
        recovery_times = []
        for _ in range(1000):
            simulated_returns = []
            current_value = 1 - abs(drawdown_pct)
            months = 0

            while current_value < 1.0 and months < 120:  # Max 10 years
                monthly_return = np.random.normal(expected_return / 12, volatility / np.sqrt(12))
                current_value *= (1 + monthly_return)
                months += 1

            recovery_times.append(months if current_value >= 1.0 else 120)

        return {
            'drawdown_pct': drawdown_pct,
            'recovery_return_needed': recovery_return_needed,
            'recovery_time_years': {
                'optimistic': optimistic_time,
                'base_case': base_time,
                'pessimistic': pessimistic_time
            },
            'monte_carlo_recovery': {
                'mean_months': np.mean(recovery_times),
                'median_months': np.median(recovery_times),
                '90th_percentile_months': np.percentile(recovery_times, 90),
                'probability_recover_1yr': np.sum(np.array(recovery_times) <= 12) / len(recovery_times),
                'probability_recover_2yr': np.sum(np.array(recovery_times) <= 24) / len(recovery_times),
                'probability_recover_5yr': np.sum(np.array(recovery_times) <= 60) / len(recovery_times)
            }
        }

    def analyze_recovery_path(self,
                             initial_portfolio: Dict[str, float],
                             drawdown_scenario: StressScenario,
                             recovery_strategy: str = 'balanced') -> Dict:
        """
        Analyze recovery path after stress scenario

        Recovery strategies:
        - 'aggressive': Higher risk/return to recover faster
        - 'balanced': Maintain current risk profile
        - 'conservative': Reduce risk to preserve capital
        """
        portfolio_value = sum(initial_portfolio.values())
        post_drawdown_value = portfolio_value * (1 + drawdown_scenario.market_shock)

        strategies = {
            'aggressive': {
                'expected_return': 0.15,
                'volatility': 0.25,
                'description': 'Increase risk to accelerate recovery'
            },
            'balanced': {
                'expected_return': 0.10,
                'volatility': 0.15,
                'description': 'Maintain current allocation'
            },
            'conservative': {
                'expected_return': 0.06,
                'volatility': 0.08,
                'description': 'Reduce risk to preserve capital'
            }
        }

        strategy_params = strategies[recovery_strategy]

        # Analyze recovery under chosen strategy
        recovery_analysis = self.analyze_recovery_time(
            drawdown_scenario.market_shock,
            strategy_params['expected_return'],
            strategy_params['volatility']
        )

        return {
            'scenario': drawdown_scenario.name,
            'initial_value': portfolio_value,
            'post_drawdown_value': post_drawdown_value,
            'recovery_strategy': recovery_strategy,
            'strategy_description': strategy_params['description'],
            'expected_return': strategy_params['expected_return'],
            'volatility': strategy_params['volatility'],
            'recovery_analysis': recovery_analysis,
            'recommendation': self._get_recovery_recommendation(recovery_analysis)
        }

    def _get_recovery_recommendation(self, recovery_analysis: Dict) -> str:
        """Generate recovery recommendation based on analysis"""
        mean_months = recovery_analysis['monte_carlo_recovery']['mean_months']
        prob_2yr = recovery_analysis['monte_carlo_recovery']['probability_recover_2yr']

        if mean_months < 12:
            return "Quick recovery expected. Maintain current strategy."
        elif mean_months < 24:
            return "Moderate recovery time. Consider balanced approach with gradual risk increase."
        elif prob_2yr > 0.7:
            return "Recovery likely within 2 years. Stay disciplined with rebalancing."
        else:
            return "Extended recovery period expected. Consider adjusting expectations or strategy."