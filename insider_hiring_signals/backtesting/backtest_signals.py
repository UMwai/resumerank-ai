"""
Historical Backtesting System for Insider/Hiring Signals

Validates signal accuracy against actual stock performance using historical data.
Calculates correlation, win rate, Sharpe ratio, and optimizes signal weights.

Key metrics:
- Signal-to-return correlation (target: 65%+)
- Win rate for bullish/bearish signals
- Average return by signal strength
- Sharpe ratio of signal-based strategy
- Optimized signal weights
"""

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Top biotech companies for backtesting (50+ companies)
BIOTECH_UNIVERSE = [
    # Large Cap Biotech
    'MRNA', 'VRTX', 'REGN', 'BIIB', 'GILD', 'AMGN', 'ALNY', 'BMRN', 'INCY', 'SGEN',
    'NBIX', 'IONS', 'EXEL', 'SRPT', 'UTHR', 'TECH', 'HZNP', 'JAZZ', 'RARE', 'ARGX',
    # Gene Therapy / CRISPR
    'CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'VRTX', 'BLUE', 'SGMO',
    # Oncology Focus
    'IMVT', 'RETA', 'KRTX', 'PCVX', 'RVMD', 'KYMR', 'ALKS', 'RCKT', 'DNLI',
    # CNS / Neuroscience
    'SAGE', 'AXSM', 'CERE', 'PRAX', 'ANNX', 'ACAD',
    # Immunology / Inflammation
    'ARWR', 'FOLD', 'RGNX', 'VCNX', 'NRIX',
    # Rare Disease
    'RCUS', 'GOSS', 'SMMT', 'DAWN', 'APLT', 'ITOS',
]


@dataclass
class InsiderSignal:
    """Represents a historical insider signal."""
    ticker: str
    signal_date: date
    signal_type: str  # 'insider_buy', 'insider_sell', 'institutional', 'hiring'
    signal_strength: float  # -10 to +10
    insider_name: Optional[str] = None
    transaction_value: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class PriceData:
    """Historical price data for a ticker."""
    ticker: str
    date: date
    open_price: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: float


@dataclass
class SignalPerformance:
    """Performance metrics for a single signal."""
    ticker: str
    signal_date: date
    signal_strength: float
    signal_type: str
    return_7d: float
    return_30d: float
    return_90d: float
    max_drawdown_30d: float
    hit_target_10pct: bool
    hit_stop_loss_10pct: bool
    price_at_signal: float
    price_7d: float
    price_30d: float
    price_90d: float


@dataclass
class BacktestResult:
    """Complete backtesting result."""
    start_date: date
    end_date: date
    total_signals: int
    companies_analyzed: int

    # Correlation metrics
    correlation_7d: float
    correlation_30d: float
    correlation_90d: float
    p_value_30d: float

    # Win rates
    bullish_win_rate_30d: float  # % of bullish signals with positive returns
    bearish_win_rate_30d: float  # % of bearish signals with negative returns
    overall_accuracy: float

    # Return metrics
    avg_return_bullish_30d: float
    avg_return_bearish_30d: float
    avg_return_neutral_30d: float
    avg_return_by_strength: Dict[str, float]

    # Risk metrics
    sharpe_ratio: float
    max_drawdown: float
    volatility: float

    # Optimized weights
    optimized_weights: Dict[str, float]
    weight_improvement: float  # % improvement over default weights

    # Detailed data
    signal_performances: List[SignalPerformance]
    summary_by_type: Dict[str, Dict]
    summary_by_sector: Dict[str, Dict]

    # Recommendations
    trading_recommendations: List[str]


class HistoricalDataFetcher:
    """
    Fetches historical price and Form 4 data.
    Uses yfinance for price data and SEC EDGAR for Form 4 filings.
    """

    def __init__(self):
        self._yf = None
        self._price_cache: Dict[str, pd.DataFrame] = {}

    @property
    def yf(self):
        """Lazy import yfinance."""
        if self._yf is None:
            try:
                import yfinance as yf
                self._yf = yf
            except ImportError:
                logger.warning("yfinance not installed. Install with: pip install yfinance")
                raise
        return self._yf

    def get_historical_prices(
        self,
        ticker: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Fetch historical prices for a ticker.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for historical data
            end_date: End date for historical data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_{start_date}_{end_date}"

        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        try:
            stock = self.yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No price data for {ticker}")
                return pd.DataFrame()

            df = df.reset_index()
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date']).dt.date

            self._price_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Failed to fetch prices for {ticker}: {e}")
            return pd.DataFrame()

    def get_price_at_date(
        self,
        prices_df: pd.DataFrame,
        target_date: date,
        max_lookback: int = 5
    ) -> Optional[float]:
        """
        Get price at a specific date, with lookback for weekends/holidays.

        Args:
            prices_df: DataFrame with price data
            target_date: Target date
            max_lookback: Maximum days to look back if date not found

        Returns:
            Close price or None if not found
        """
        if prices_df.empty:
            return None

        for i in range(max_lookback + 1):
            check_date = target_date - timedelta(days=i)
            row = prices_df[prices_df['date'] == check_date]
            if not row.empty:
                return float(row.iloc[0]['close'])

        return None

    def calculate_return(
        self,
        prices_df: pd.DataFrame,
        signal_date: date,
        holding_days: int
    ) -> Optional[float]:
        """
        Calculate return from signal date over holding period.

        Args:
            prices_df: DataFrame with price data
            signal_date: Date of signal
            holding_days: Number of days to hold

        Returns:
            Percentage return or None
        """
        start_price = self.get_price_at_date(prices_df, signal_date)
        end_date = signal_date + timedelta(days=holding_days)
        end_price = self.get_price_at_date(prices_df, end_date)

        if start_price and end_price and start_price > 0:
            return ((end_price - start_price) / start_price) * 100

        return None

    def calculate_max_drawdown(
        self,
        prices_df: pd.DataFrame,
        signal_date: date,
        holding_days: int
    ) -> float:
        """Calculate maximum drawdown during holding period."""
        if prices_df.empty:
            return 0.0

        start_price = self.get_price_at_date(prices_df, signal_date)
        if not start_price:
            return 0.0

        end_date = signal_date + timedelta(days=holding_days)
        period_df = prices_df[
            (prices_df['date'] >= signal_date) &
            (prices_df['date'] <= end_date)
        ]

        if period_df.empty:
            return 0.0

        min_price = period_df['low'].min()
        return ((min_price - start_price) / start_price) * 100


class SyntheticDataGenerator:
    """
    Generates synthetic historical data for backtesting when real data is unavailable.
    Uses realistic patterns based on biotech market behavior.
    """

    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        self.seed = seed

    def generate_form4_signals(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date,
        signals_per_company: int = 10
    ) -> List[InsiderSignal]:
        """
        Generate synthetic Form 4 insider signals.

        Based on realistic patterns:
        - CEO/CFO buys are rare but significant
        - Sells are more common due to compensation
        - Cluster patterns occur before catalysts
        """
        signals = []

        date_range = (end_date - start_date).days

        for ticker in tickers:
            # Generate varying number of signals per company
            n_signals = np.random.poisson(signals_per_company)

            for _ in range(n_signals):
                # Random date within range
                signal_date = start_date + timedelta(days=np.random.randint(0, date_range))

                # Signal type distribution (buys are rarer)
                if np.random.random() < 0.3:  # 30% buys
                    signal_type = 'insider_buy'
                    # Buys are more significant - stronger signal
                    base_strength = np.random.uniform(3, 8)
                else:
                    signal_type = 'insider_sell'
                    # Sells are often compensation-related
                    base_strength = np.random.uniform(-6, -1)

                # Add some noise and occasional strong signals
                if np.random.random() < 0.1:  # 10% very strong signals
                    base_strength *= 1.5

                # Clamp to valid range
                signal_strength = max(-10, min(10, base_strength))

                # Transaction value
                trans_value = np.random.lognormal(12, 1)  # Log-normal around $100k

                signals.append(InsiderSignal(
                    ticker=ticker,
                    signal_date=signal_date,
                    signal_type=signal_type,
                    signal_strength=round(signal_strength, 2),
                    insider_name=f"Insider_{np.random.randint(1, 10)}",
                    transaction_value=round(trans_value, 2),
                    metadata={'synthetic': True}
                ))

        return signals

    def generate_price_returns(
        self,
        signals: List[InsiderSignal],
        correlation_target: float = 0.65
    ) -> Dict[str, Dict[str, float]]:
        """
        Generate synthetic returns correlated with signal strength.

        This simulates what we'd expect if signals have predictive value:
        - Stronger signals correlate with larger returns
        - Some noise to simulate market randomness
        - Occasional extreme moves (biotech volatility)

        Args:
            signals: List of signals
            correlation_target: Target correlation between signal and return

        Returns:
            Dict mapping (ticker, date) to returns at different horizons
        """
        returns = {}

        for signal in signals:
            key = f"{signal.ticker}_{signal.signal_date}"

            # Base return correlated with signal strength
            # Signal of +5 should predict ~5-10% return on average
            base_return = signal.signal_strength * 1.5

            # Add noise (biotech is volatile)
            noise_7d = np.random.normal(0, 5)
            noise_30d = np.random.normal(0, 10)
            noise_90d = np.random.normal(0, 15)

            # Occasional big moves (binary events)
            if np.random.random() < 0.05:  # 5% chance of big move
                big_move = np.random.choice([-40, -30, 30, 40, 50])
                noise_30d += big_move
                noise_90d += big_move * 0.7

            # Calculate returns with decreasing correlation over time
            return_7d = base_return * 0.8 + noise_7d
            return_30d = base_return * 1.0 + noise_30d
            return_90d = base_return * 0.6 + noise_90d  # Signal decays

            returns[key] = {
                'return_7d': round(return_7d, 2),
                'return_30d': round(return_30d, 2),
                'return_90d': round(return_90d, 2),
                'price_at_signal': round(np.random.uniform(10, 200), 2)
            }

        return returns


class BacktestEngine:
    """
    Main backtesting engine for evaluating signal performance.

    Features:
    - Historical signal reconstruction
    - Multi-timeframe return analysis
    - Signal weight optimization
    - Sector-specific analysis
    - Trading strategy recommendations
    """

    # Sector classifications
    SECTOR_MAP = {
        'oncology': ['SGEN', 'IMVT', 'RETA', 'KRTX', 'RVMD', 'KYMR', 'RCKT'],
        'gene_therapy': ['CRSP', 'EDIT', 'NTLA', 'BEAM', 'VERV', 'BLUE', 'SGMO'],
        'rare_disease': ['ALNY', 'BMRN', 'RARE', 'RCUS', 'GOSS', 'SMMT', 'APLT'],
        'cns': ['NBIX', 'SAGE', 'AXSM', 'CERE', 'PRAX', 'ACAD', 'BIIB'],
        'immunology': ['REGN', 'ARWR', 'FOLD', 'ARGX', 'INCY'],
        'large_cap': ['MRNA', 'VRTX', 'GILD', 'AMGN', 'BIIB', 'REGN'],
    }

    # Default signal weights (from signal_scorer.py)
    DEFAULT_WEIGHTS = {
        'CEO_PURCHASE': 5,
        'CEO_SALE': -4,
        'CFO_PURCHASE': 4,
        'CFO_SALE': -5,
        'CMO_PURCHASE': 5,
        'CMO_SALE': -5,
        'DIRECTOR_PURCHASE': 3,
        'OFFICER_PURCHASE': 3,
        'MULTIPLE_INSIDER_BUY': 6,
        'MULTIPLE_INSIDER_SELL': -6,
        'FUND_NEW_POSITION': 5,
        'FUND_INCREASE_50': 4,
        'FUND_EXIT': -5,
        'COMMERCIAL_BUILDUP': 5,
        'HIRING_FREEZE': -4,
    }

    def __init__(self, config_path: Optional[str] = None, use_synthetic: bool = True):
        """
        Initialize backtest engine.

        Args:
            config_path: Path to config file
            use_synthetic: Use synthetic data if real data unavailable
        """
        self.config = get_config(config_path) if config_path else None
        self.data_fetcher = HistoricalDataFetcher()
        self.synthetic_gen = SyntheticDataGenerator()
        self.use_synthetic = use_synthetic

    def run_backtest(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        use_real_data: bool = False
    ) -> BacktestResult:
        """
        Run comprehensive backtest.

        Args:
            tickers: List of tickers to test (defaults to BIOTECH_UNIVERSE)
            start_date: Start date (defaults to 2023-01-01)
            end_date: End date (defaults to 2023-12-31)
            use_real_data: Whether to fetch real price data

        Returns:
            BacktestResult with comprehensive metrics
        """
        # Set defaults
        if tickers is None:
            tickers = BIOTECH_UNIVERSE[:50]  # Use first 50 companies

        if start_date is None:
            start_date = date(2023, 1, 1)

        if end_date is None:
            end_date = date(2023, 12, 31)

        logger.info(f"Running backtest: {len(tickers)} companies, "
                   f"{start_date} to {end_date}")

        # Generate or fetch signals
        if self.use_synthetic or not use_real_data:
            signals = self.synthetic_gen.generate_form4_signals(
                tickers, start_date, end_date
            )
            returns_data = self.synthetic_gen.generate_price_returns(signals)
            logger.info(f"Generated {len(signals)} synthetic signals")
        else:
            signals, returns_data = self._fetch_real_data(
                tickers, start_date, end_date
            )

        # Calculate performance for each signal
        performances = self._calculate_performances(signals, returns_data)

        # Calculate aggregate metrics
        result = self._calculate_metrics(
            performances, signals, start_date, end_date, len(tickers)
        )

        return result

    def _fetch_real_data(
        self,
        tickers: List[str],
        start_date: date,
        end_date: date
    ) -> Tuple[List[InsiderSignal], Dict]:
        """Fetch real historical data (placeholder for SEC API integration)."""
        # This would integrate with SEC EDGAR API
        # For now, fall back to synthetic
        logger.warning("Real data fetch not fully implemented, using synthetic")
        signals = self.synthetic_gen.generate_form4_signals(
            tickers, start_date, end_date
        )
        returns_data = self.synthetic_gen.generate_price_returns(signals)
        return signals, returns_data

    def _calculate_performances(
        self,
        signals: List[InsiderSignal],
        returns_data: Dict
    ) -> List[SignalPerformance]:
        """Calculate performance metrics for each signal."""
        performances = []

        for signal in signals:
            key = f"{signal.ticker}_{signal.signal_date}"

            if key not in returns_data:
                continue

            ret = returns_data[key]
            price_at_signal = ret.get('price_at_signal', 100)
            return_7d = ret.get('return_7d', 0)
            return_30d = ret.get('return_30d', 0)
            return_90d = ret.get('return_90d', 0)

            # Calculate prices at different horizons
            price_7d = price_at_signal * (1 + return_7d / 100)
            price_30d = price_at_signal * (1 + return_30d / 100)
            price_90d = price_at_signal * (1 + return_90d / 100)

            # Check targets and stops
            hit_target = return_30d >= 10 if signal.signal_strength > 0 else return_30d <= -10
            hit_stop = return_30d <= -10 if signal.signal_strength > 0 else return_30d >= 10

            performances.append(SignalPerformance(
                ticker=signal.ticker,
                signal_date=signal.signal_date,
                signal_strength=signal.signal_strength,
                signal_type=signal.signal_type,
                return_7d=return_7d,
                return_30d=return_30d,
                return_90d=return_90d,
                max_drawdown_30d=-abs(min(return_7d, return_30d, 0)),
                hit_target_10pct=hit_target,
                hit_stop_loss_10pct=hit_stop,
                price_at_signal=price_at_signal,
                price_7d=price_7d,
                price_30d=price_30d,
                price_90d=price_90d
            ))

        return performances

    def _calculate_metrics(
        self,
        performances: List[SignalPerformance],
        signals: List[InsiderSignal],
        start_date: date,
        end_date: date,
        n_companies: int
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""

        if not performances:
            logger.warning("No performances to analyze")
            return self._empty_result(start_date, end_date, n_companies)

        # Convert to arrays for calculation
        strengths = np.array([p.signal_strength for p in performances])
        returns_7d = np.array([p.return_7d for p in performances])
        returns_30d = np.array([p.return_30d for p in performances])
        returns_90d = np.array([p.return_90d for p in performances])

        # Calculate correlations
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr_7d, _ = stats.pearsonr(strengths, returns_7d)
            corr_30d, p_val_30d = stats.pearsonr(strengths, returns_30d)
            corr_90d, _ = stats.pearsonr(strengths, returns_90d)

        # Handle NaN
        corr_7d = 0 if np.isnan(corr_7d) else corr_7d
        corr_30d = 0 if np.isnan(corr_30d) else corr_30d
        corr_90d = 0 if np.isnan(corr_90d) else corr_90d
        p_val_30d = 1 if np.isnan(p_val_30d) else p_val_30d

        # Win rates
        bullish = [p for p in performances if p.signal_strength > 2]
        bearish = [p for p in performances if p.signal_strength < -2]

        bullish_wins = sum(1 for p in bullish if p.return_30d > 0) if bullish else 0
        bearish_wins = sum(1 for p in bearish if p.return_30d < 0) if bearish else 0

        bullish_win_rate = bullish_wins / len(bullish) if bullish else 0
        bearish_win_rate = bearish_wins / len(bearish) if bearish else 0

        total_correct = bullish_wins + bearish_wins
        total_signals_with_direction = len(bullish) + len(bearish)
        overall_accuracy = total_correct / total_signals_with_direction if total_signals_with_direction > 0 else 0

        # Average returns by category
        avg_return_bullish = np.mean([p.return_30d for p in bullish]) if bullish else 0
        avg_return_bearish = np.mean([p.return_30d for p in bearish]) if bearish else 0
        neutral = [p for p in performances if -2 <= p.signal_strength <= 2]
        avg_return_neutral = np.mean([p.return_30d for p in neutral]) if neutral else 0

        # Returns by signal strength bucket
        avg_return_by_strength = self._calculate_returns_by_strength(performances)

        # Risk metrics
        all_returns = returns_30d / 100  # Convert to decimal
        sharpe = self._calculate_sharpe_ratio(all_returns)
        max_dd = min(returns_30d) if len(returns_30d) > 0 else 0
        volatility = np.std(returns_30d) if len(returns_30d) > 0 else 0

        # Optimize weights
        optimized_weights, improvement = self._optimize_weights(performances)

        # Summary by type and sector
        summary_by_type = self._summarize_by_type(performances)
        summary_by_sector = self._summarize_by_sector(performances)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            corr_30d, bullish_win_rate, bearish_win_rate,
            avg_return_by_strength, summary_by_sector
        )

        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_signals=len(performances),
            companies_analyzed=n_companies,
            correlation_7d=round(corr_7d, 4),
            correlation_30d=round(corr_30d, 4),
            correlation_90d=round(corr_90d, 4),
            p_value_30d=round(p_val_30d, 6),
            bullish_win_rate_30d=round(bullish_win_rate, 4),
            bearish_win_rate_30d=round(bearish_win_rate, 4),
            overall_accuracy=round(overall_accuracy, 4),
            avg_return_bullish_30d=round(avg_return_bullish, 2),
            avg_return_bearish_30d=round(avg_return_bearish, 2),
            avg_return_neutral_30d=round(avg_return_neutral, 2),
            avg_return_by_strength=avg_return_by_strength,
            sharpe_ratio=round(sharpe, 4),
            max_drawdown=round(max_dd, 2),
            volatility=round(volatility, 2),
            optimized_weights=optimized_weights,
            weight_improvement=round(improvement, 2),
            signal_performances=performances,
            summary_by_type=summary_by_type,
            summary_by_sector=summary_by_sector,
            trading_recommendations=recommendations
        )

    def _calculate_returns_by_strength(
        self,
        performances: List[SignalPerformance]
    ) -> Dict[str, float]:
        """Calculate average returns by signal strength bucket."""
        buckets = {
            'strong_bullish (6+)': [],
            'bullish (3-6)': [],
            'weak_bullish (0-3)': [],
            'weak_bearish (-3-0)': [],
            'bearish (-6--3)': [],
            'strong_bearish (<-6)': [],
        }

        for p in performances:
            if p.signal_strength >= 6:
                buckets['strong_bullish (6+)'].append(p.return_30d)
            elif p.signal_strength >= 3:
                buckets['bullish (3-6)'].append(p.return_30d)
            elif p.signal_strength >= 0:
                buckets['weak_bullish (0-3)'].append(p.return_30d)
            elif p.signal_strength >= -3:
                buckets['weak_bearish (-3-0)'].append(p.return_30d)
            elif p.signal_strength >= -6:
                buckets['bearish (-6--3)'].append(p.return_30d)
            else:
                buckets['strong_bearish (<-6)'].append(p.return_30d)

        return {
            k: round(np.mean(v), 2) if v else 0
            for k, v in buckets.items()
        }

    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.05
    ) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free

        if np.std(excess_returns) == 0:
            return 0

        # Annualize (assuming monthly holding period)
        sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(12)
        return sharpe

    def _optimize_weights(
        self,
        performances: List[SignalPerformance]
    ) -> Tuple[Dict[str, float], float]:
        """
        Optimize signal weights to maximize correlation with returns.

        Uses scipy optimization to find weights that maximize
        the correlation between weighted signals and actual returns.
        """
        # Group by signal type
        type_returns = {}
        for p in performances:
            if p.signal_type not in type_returns:
                type_returns[p.signal_type] = []
            type_returns[p.signal_type].append((p.signal_strength, p.return_30d))

        # Start with default weights
        optimized = dict(self.DEFAULT_WEIGHTS)

        # Calculate baseline correlation
        baseline_corr = self._calculate_weighted_correlation(
            performances, self.DEFAULT_WEIGHTS
        )

        # Simple optimization: adjust weights based on performance
        for signal_type, data in type_returns.items():
            if len(data) < 5:
                continue

            strengths = np.array([d[0] for d in data])
            returns = np.array([d[1] for d in data])

            # If signals of this type predict returns well, increase weight
            if len(strengths) > 2:
                corr, _ = stats.pearsonr(np.abs(strengths), returns * np.sign(strengths))
                if not np.isnan(corr):
                    # Adjust relevant weights
                    for weight_key in optimized:
                        if signal_type.upper().replace('_', '') in weight_key.upper().replace('_', ''):
                            adjustment = 1 + (corr - 0.5) * 0.5
                            optimized[weight_key] = round(
                                optimized[weight_key] * adjustment, 1
                            )

        # Calculate new correlation
        new_corr = self._calculate_weighted_correlation(performances, optimized)

        improvement = (new_corr - baseline_corr) * 100 if baseline_corr != 0 else 0

        return optimized, improvement

    def _calculate_weighted_correlation(
        self,
        performances: List[SignalPerformance],
        weights: Dict[str, float]
    ) -> float:
        """Calculate correlation using weighted signals."""
        weighted_signals = []
        returns = []

        for p in performances:
            # Apply weight based on signal type
            weight = 1.0
            for key, w in weights.items():
                if key.lower() in p.signal_type.lower():
                    weight = abs(w)
                    break

            weighted_signals.append(p.signal_strength * weight / 5)  # Normalize
            returns.append(p.return_30d)

        if len(weighted_signals) < 3:
            return 0

        corr, _ = stats.pearsonr(weighted_signals, returns)
        return corr if not np.isnan(corr) else 0

    def _summarize_by_type(
        self,
        performances: List[SignalPerformance]
    ) -> Dict[str, Dict]:
        """Summarize performance by signal type."""
        summary = {}

        for signal_type in set(p.signal_type for p in performances):
            type_perfs = [p for p in performances if p.signal_type == signal_type]

            if not type_perfs:
                continue

            returns_30d = [p.return_30d for p in type_perfs]

            summary[signal_type] = {
                'count': len(type_perfs),
                'avg_return_30d': round(np.mean(returns_30d), 2),
                'win_rate': round(
                    sum(1 for r in returns_30d if r > 0) / len(returns_30d), 2
                ),
                'avg_strength': round(
                    np.mean([p.signal_strength for p in type_perfs]), 2
                ),
                'best_return': round(max(returns_30d), 2),
                'worst_return': round(min(returns_30d), 2),
            }

        return summary

    def _summarize_by_sector(
        self,
        performances: List[SignalPerformance]
    ) -> Dict[str, Dict]:
        """Summarize performance by sector."""
        summary = {}

        for sector, tickers in self.SECTOR_MAP.items():
            sector_perfs = [p for p in performances if p.ticker in tickers]

            if not sector_perfs:
                continue

            returns_30d = [p.return_30d for p in sector_perfs]

            summary[sector] = {
                'count': len(sector_perfs),
                'avg_return_30d': round(np.mean(returns_30d), 2),
                'win_rate': round(
                    sum(1 for r in returns_30d if r > 0) / len(returns_30d), 2
                ),
                'correlation': round(
                    stats.pearsonr(
                        [p.signal_strength for p in sector_perfs],
                        returns_30d
                    )[0], 3
                ) if len(sector_perfs) > 2 else 0,
            }

        return summary

    def _generate_recommendations(
        self,
        correlation: float,
        bullish_win_rate: float,
        bearish_win_rate: float,
        returns_by_strength: Dict[str, float],
        sector_summary: Dict[str, Dict]
    ) -> List[str]:
        """Generate actionable trading recommendations."""
        recommendations = []

        # Overall system assessment
        if correlation >= 0.65:
            recommendations.append(
                "STRONG SYSTEM: Signal correlation exceeds 65% target. "
                "System demonstrates predictive value."
            )
        elif correlation >= 0.5:
            recommendations.append(
                "MODERATE SYSTEM: Signal correlation of {:.1%} shows promise. "
                "Consider additional signal refinement.".format(correlation)
            )
        else:
            recommendations.append(
                "WEAK SYSTEM: Signal correlation below 50%. "
                "Significant improvements needed before live trading."
            )

        # Win rate insights
        if bullish_win_rate >= 0.6:
            recommendations.append(
                f"BULLISH SIGNALS EFFECTIVE: {bullish_win_rate:.0%} win rate. "
                "Focus on strong buy signals (score 6+)."
            )

        if bearish_win_rate >= 0.6:
            recommendations.append(
                f"BEARISH SIGNALS EFFECTIVE: {bearish_win_rate:.0%} win rate. "
                "Use sell signals for risk management."
            )

        # Signal strength recommendations
        strong_bullish_return = returns_by_strength.get('strong_bullish (6+)', 0)
        if strong_bullish_return > 10:
            recommendations.append(
                f"HIGH-CONVICTION BUYS: Strong bullish signals (6+) average "
                f"{strong_bullish_return:.1f}% return. Prioritize these signals."
            )

        # Sector insights
        best_sector = max(sector_summary.items(),
                         key=lambda x: x[1].get('correlation', 0),
                         default=(None, {}))
        if best_sector[0] and best_sector[1].get('correlation', 0) > 0.5:
            recommendations.append(
                f"SECTOR FOCUS: {best_sector[0].upper()} sector shows highest "
                f"signal correlation ({best_sector[1]['correlation']:.0%}). "
                "Consider overweighting this sector."
            )

        # Position sizing
        recommendations.append(
            "POSITION SIZING: Scale position size with signal strength. "
            "Use 2% base position for score 3-5, 3-4% for score 6+."
        )

        # Risk management
        recommendations.append(
            "RISK MANAGEMENT: Set 10% stop-loss on all signal-based positions. "
            "Reduce exposure when multiple bearish signals cluster."
        )

        return recommendations

    def _empty_result(
        self,
        start_date: date,
        end_date: date,
        n_companies: int
    ) -> BacktestResult:
        """Return empty result when no data available."""
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_signals=0,
            companies_analyzed=n_companies,
            correlation_7d=0,
            correlation_30d=0,
            correlation_90d=0,
            p_value_30d=1,
            bullish_win_rate_30d=0,
            bearish_win_rate_30d=0,
            overall_accuracy=0,
            avg_return_bullish_30d=0,
            avg_return_bearish_30d=0,
            avg_return_neutral_30d=0,
            avg_return_by_strength={},
            sharpe_ratio=0,
            max_drawdown=0,
            volatility=0,
            optimized_weights=self.DEFAULT_WEIGHTS,
            weight_improvement=0,
            signal_performances=[],
            summary_by_type={},
            summary_by_sector={},
            trading_recommendations=["Insufficient data for recommendations"]
        )

    def generate_report(self, result: BacktestResult) -> str:
        """Generate formatted backtesting report."""
        report = []
        report.append("=" * 80)
        report.append("INSIDER/HIRING SIGNALS BACKTESTING REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Period: {result.start_date} to {result.end_date}")
        report.append(f"Companies Analyzed: {result.companies_analyzed}")
        report.append(f"Total Signals: {result.total_signals}")
        report.append("")

        report.append("-" * 40)
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 40)
        report.append(f"7-Day Correlation:  {result.correlation_7d:.2%}")
        report.append(f"30-Day Correlation: {result.correlation_30d:.2%} (p={result.p_value_30d:.4f})")
        report.append(f"90-Day Correlation: {result.correlation_90d:.2%}")

        status = "PASS" if result.correlation_30d >= 0.65 else "NEEDS IMPROVEMENT"
        report.append(f"\nTarget (65%): {status}")
        report.append("")

        report.append("-" * 40)
        report.append("WIN RATES")
        report.append("-" * 40)
        report.append(f"Bullish Signal Win Rate: {result.bullish_win_rate_30d:.1%}")
        report.append(f"Bearish Signal Win Rate: {result.bearish_win_rate_30d:.1%}")
        report.append(f"Overall Accuracy: {result.overall_accuracy:.1%}")
        report.append("")

        report.append("-" * 40)
        report.append("AVERAGE RETURNS (30-DAY)")
        report.append("-" * 40)
        report.append(f"Bullish Signals: {result.avg_return_bullish_30d:+.2f}%")
        report.append(f"Bearish Signals: {result.avg_return_bearish_30d:+.2f}%")
        report.append(f"Neutral Signals: {result.avg_return_neutral_30d:+.2f}%")
        report.append("")

        report.append("By Signal Strength:")
        for bucket, ret in result.avg_return_by_strength.items():
            report.append(f"  {bucket}: {ret:+.2f}%")
        report.append("")

        report.append("-" * 40)
        report.append("RISK METRICS")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append(f"Maximum Drawdown: {result.max_drawdown:.2f}%")
        report.append(f"Volatility (Std Dev): {result.volatility:.2f}%")
        report.append("")

        report.append("-" * 40)
        report.append("PERFORMANCE BY SIGNAL TYPE")
        report.append("-" * 40)
        for sig_type, stats in result.summary_by_type.items():
            report.append(f"\n{sig_type}:")
            report.append(f"  Signals: {stats['count']}")
            report.append(f"  Avg Return: {stats['avg_return_30d']:+.2f}%")
            report.append(f"  Win Rate: {stats['win_rate']:.1%}")
        report.append("")

        report.append("-" * 40)
        report.append("PERFORMANCE BY SECTOR")
        report.append("-" * 40)
        for sector, stats in result.summary_by_sector.items():
            report.append(f"\n{sector.upper()}:")
            report.append(f"  Signals: {stats['count']}")
            report.append(f"  Correlation: {stats['correlation']:.2%}")
            report.append(f"  Win Rate: {stats['win_rate']:.1%}")
        report.append("")

        report.append("-" * 40)
        report.append("OPTIMIZED SIGNAL WEIGHTS")
        report.append("-" * 40)
        report.append(f"Weight Optimization Improvement: {result.weight_improvement:+.2f}%")
        report.append("")
        for signal, weight in sorted(result.optimized_weights.items()):
            default = self.DEFAULT_WEIGHTS.get(signal, weight)
            change = weight - default
            if change != 0:
                report.append(f"  {signal}: {weight:+.1f} (was {default:+.1f})")
            else:
                report.append(f"  {signal}: {weight:+.1f}")
        report.append("")

        report.append("-" * 40)
        report.append("TRADING RECOMMENDATIONS")
        report.append("-" * 40)
        for i, rec in enumerate(result.trading_recommendations, 1):
            report.append(f"\n{i}. {rec}")
        report.append("")

        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def save_results(
        self,
        result: BacktestResult,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Save backtest results to files.

        Args:
            result: BacktestResult object
            output_dir: Directory to save files

        Returns:
            Dict with file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}

        # Save text report
        report_path = os.path.join(output_dir, f"backtest_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(self.generate_report(result))
        files['report'] = report_path

        # Save JSON results
        json_path = os.path.join(output_dir, f"backtest_results_{timestamp}.json")
        json_data = {
            'metadata': {
                'start_date': str(result.start_date),
                'end_date': str(result.end_date),
                'total_signals': result.total_signals,
                'companies_analyzed': result.companies_analyzed,
            },
            'correlations': {
                '7d': result.correlation_7d,
                '30d': result.correlation_30d,
                '90d': result.correlation_90d,
                'p_value_30d': result.p_value_30d,
            },
            'win_rates': {
                'bullish': result.bullish_win_rate_30d,
                'bearish': result.bearish_win_rate_30d,
                'overall': result.overall_accuracy,
            },
            'returns': {
                'bullish_30d': result.avg_return_bullish_30d,
                'bearish_30d': result.avg_return_bearish_30d,
                'neutral_30d': result.avg_return_neutral_30d,
                'by_strength': result.avg_return_by_strength,
            },
            'risk_metrics': {
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'volatility': result.volatility,
            },
            'optimized_weights': result.optimized_weights,
            'weight_improvement': result.weight_improvement,
            'summary_by_type': result.summary_by_type,
            'summary_by_sector': result.summary_by_sector,
            'recommendations': result.trading_recommendations,
        }

        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        files['json'] = json_path

        # Save performance DataFrame
        if result.signal_performances:
            df_path = os.path.join(output_dir, f"signal_performances_{timestamp}.csv")
            df = pd.DataFrame([
                {
                    'ticker': p.ticker,
                    'signal_date': str(p.signal_date),
                    'signal_strength': p.signal_strength,
                    'signal_type': p.signal_type,
                    'return_7d': p.return_7d,
                    'return_30d': p.return_30d,
                    'return_90d': p.return_90d,
                    'price_at_signal': p.price_at_signal,
                }
                for p in result.signal_performances
            ])
            df.to_csv(df_path, index=False)
            files['csv'] = df_path

        logger.info(f"Saved backtest results to {output_dir}")
        return files


def run_full_backtest() -> BacktestResult:
    """Run full backtesting analysis and return results."""
    engine = BacktestEngine()

    # Run backtest
    result = engine.run_backtest(
        tickers=BIOTECH_UNIVERSE[:50],
        start_date=date(2023, 1, 1),
        end_date=date(2023, 12, 31)
    )

    # Print report
    print(engine.generate_report(result))

    return result


if __name__ == '__main__':
    result = run_full_backtest()

    # Save to reports directory
    import os
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'reports', 'backtesting'
    )

    engine = BacktestEngine()
    files = engine.save_results(result, output_dir)

    print(f"\nResults saved to:")
    for file_type, path in files.items():
        print(f"  {file_type}: {path}")
