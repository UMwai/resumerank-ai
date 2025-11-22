"""
Simple Data Collector Module
============================
Simplified data collection using Yahoo Finance only.
No Kafka, no complex multi-source integration.
Batch processing with 15-minute intervals.

Author: Backend Systems Architect
Date: November 2024
Version: 1.0
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Union
import os
import json
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleDataCollector:
    """
    Simplified data collector for Yahoo Finance data.
    Focuses on batch processing without real-time streaming complexity.
    """

    def __init__(self, data_dir: str = "/Users/waiyang/Desktop/repo/dreamers-v2/data"):
        """
        Initialize data collector with data directory.

        Args:
            data_dir: Directory to save downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.historical_dir = self.data_dir / "historical"
        self.realtime_dir = self.data_dir / "realtime"
        self.historical_dir.mkdir(exist_ok=True)
        self.realtime_dir.mkdir(exist_ok=True)

        # Data manifest to track available symbols
        self.manifest_file = self.data_dir / "manifest.json"
        self.manifest = self._load_manifest()

        logger.info(f"Data collector initialized. Data directory: {self.data_dir}")

    def fetch_historical(
        self,
        symbols: Union[str, List[str]],
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for given symbols.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format), defaults to today
            interval: Data interval (1d, 1h, 15m, etc.)

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        results = {}
        failed_symbols = []

        for symbol in symbols:
            try:
                logger.info(f"Fetching historical data for {symbol}...")

                # Download data using yfinance
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )

                if data.empty:
                    logger.warning(f"No data returned for {symbol}")
                    failed_symbols.append(symbol)
                    continue

                # Clean up column names
                data.columns = [col.lower() for col in data.columns]

                # Add additional calculated fields
                data['returns'] = data['close'].pct_change()
                data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['sma_50'] = data['close'].rolling(window=50).mean()
                data['sma_200'] = data['close'].rolling(window=200).mean()

                # Calculate volatility
                data['volatility_20d'] = data['returns'].rolling(window=20).std() * np.sqrt(252)

                # Store in results
                results[symbol] = data

                # Update manifest
                self._update_manifest(symbol, start_date, end_date, len(data))

                logger.info(f"Successfully fetched {len(data)} records for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                failed_symbols.append(symbol)

        if failed_symbols:
            logger.warning(f"Failed to fetch data for: {failed_symbols}")

        return results

    def fetch_realtime(
        self,
        symbols: Union[str, List[str]],
        period: str = "1d",
        interval: str = "15m"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch near real-time data (15-minute intervals).

        Args:
            symbols: Single symbol or list of symbols
            period: Period to fetch (1d, 5d, 1mo, etc.)
            interval: Data interval (1m, 5m, 15m, 1h)

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        results = {}

        for symbol in symbols:
            try:
                logger.info(f"Fetching realtime data for {symbol}...")

                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period, interval=interval)

                if data.empty:
                    logger.warning(f"No realtime data for {symbol}")
                    continue

                # Clean column names
                data.columns = [col.lower() for col in data.columns]

                # Add calculated fields
                data['returns'] = data['close'].pct_change()
                data['vwap'] = (data['volume'] * data['close']).cumsum() / data['volume'].cumsum()

                # Calculate intraday metrics
                data['intraday_range'] = (data['high'] - data['low']) / data['open'] * 100
                data['close_to_high'] = (data['close'] - data['low']) / (data['high'] - data['low'])

                results[symbol] = data

                # Save latest data with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self._save_realtime_snapshot(symbol, data, timestamp)

                logger.info(f"Fetched {len(data)} realtime records for {symbol}")

            except Exception as e:
                logger.error(f"Error fetching realtime data for {symbol}: {str(e)}")

        return results

    def save_to_csv(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        filename: Optional[str] = None,
        subfolder: str = "historical"
    ) -> List[str]:
        """
        Save data to CSV files.

        Args:
            data: DataFrame or dictionary of DataFrames
            filename: Optional filename (auto-generated if not provided)
            subfolder: Subfolder to save in (historical or realtime)

        Returns:
            List of saved file paths
        """
        saved_files = []
        target_dir = self.data_dir / subfolder
        target_dir.mkdir(exist_ok=True)

        if isinstance(data, pd.DataFrame):
            # Single DataFrame
            if filename is None:
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

            filepath = target_dir / filename
            data.to_csv(filepath)
            saved_files.append(str(filepath))
            logger.info(f"Saved data to {filepath}")

        elif isinstance(data, dict):
            # Dictionary of DataFrames
            for symbol, df in data.items():
                if filename:
                    file = f"{symbol}_{filename}"
                else:
                    file = f"{symbol}_{datetime.now().strftime('%Y%m%d')}.csv"

                filepath = target_dir / file
                df.to_csv(filepath)
                saved_files.append(str(filepath))
                logger.info(f"Saved {symbol} data to {filepath}")

        return saved_files

    def load_from_csv(self, symbol: str, subfolder: str = "historical") -> Optional[pd.DataFrame]:
        """
        Load data from CSV file.

        Args:
            symbol: Symbol to load
            subfolder: Subfolder to load from

        Returns:
            DataFrame or None if not found
        """
        target_dir = self.data_dir / subfolder

        # Find most recent file for symbol
        matching_files = list(target_dir.glob(f"{symbol}_*.csv"))

        if not matching_files:
            logger.warning(f"No data files found for {symbol}")
            return None

        # Sort by modification time and get most recent
        latest_file = max(matching_files, key=lambda f: f.stat().st_mtime)

        logger.info(f"Loading data from {latest_file}")
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)

        return df

    def get_vix_data(
        self,
        start_date: str = "2019-01-01",
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch VIX (volatility index) data.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            VIX DataFrame
        """
        vix_data = self.fetch_historical("^VIX", start_date, end_date)

        if "^VIX" in vix_data:
            vix_df = vix_data["^VIX"]

            # Add VIX-specific calculations
            vix_df['vix_ma20'] = vix_df['close'].rolling(window=20).mean()
            vix_df['vix_percentile'] = vix_df['close'].rolling(window=252).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
            )

            # VIX regime levels
            vix_df['vix_regime'] = pd.cut(
                vix_df['close'],
                bins=[0, 15, 20, 25, 30, 100],
                labels=['very_low', 'low', 'medium', 'high', 'extreme']
            )

            return vix_df

        return pd.DataFrame()

    def get_market_breadth(
        self,
        symbols: List[str],
        date: Optional[datetime] = None
    ) -> Dict[str, float]:
        """
        Calculate market breadth indicators.

        Args:
            symbols: List of symbols to analyze
            date: Date to calculate breadth (defaults to latest)

        Returns:
            Dictionary of breadth indicators
        """
        breadth = {
            'advancing': 0,
            'declining': 0,
            'unchanged': 0,
            'above_sma50': 0,
            'above_sma200': 0,
            'new_highs': 0,
            'new_lows': 0
        }

        for symbol in symbols:
            try:
                data = self.load_from_csv(symbol)

                if data is None or data.empty:
                    continue

                # Get latest or specific date data
                if date:
                    if date in data.index:
                        row = data.loc[date]
                        prev_row = data.loc[:date].iloc[-2] if len(data.loc[:date]) > 1 else None
                    else:
                        continue
                else:
                    row = data.iloc[-1]
                    prev_row = data.iloc[-2] if len(data) > 1 else None

                if prev_row is not None:
                    # Count advancing/declining
                    if row['close'] > prev_row['close']:
                        breadth['advancing'] += 1
                    elif row['close'] < prev_row['close']:
                        breadth['declining'] += 1
                    else:
                        breadth['unchanged'] += 1

                # Check SMA positions
                if 'sma_50' in row and pd.notna(row['sma_50']):
                    if row['close'] > row['sma_50']:
                        breadth['above_sma50'] += 1

                if 'sma_200' in row and pd.notna(row['sma_200']):
                    if row['close'] > row['sma_200']:
                        breadth['above_sma200'] += 1

                # Check for new highs/lows (52-week)
                if len(data) >= 252:
                    yearly_data = data.iloc[-252:]
                    if row['close'] == yearly_data['close'].max():
                        breadth['new_highs'] += 1
                    elif row['close'] == yearly_data['close'].min():
                        breadth['new_lows'] += 1

            except Exception as e:
                logger.error(f"Error calculating breadth for {symbol}: {str(e)}")
                continue

        # Calculate ratios
        total = len(symbols)
        if total > 0:
            breadth['advance_decline_ratio'] = (
                breadth['advancing'] / breadth['declining']
                if breadth['declining'] > 0 else float('inf')
            )
            breadth['pct_above_sma50'] = breadth['above_sma50'] / total
            breadth['pct_above_sma200'] = breadth['above_sma200'] / total
            breadth['high_low_ratio'] = (
                breadth['new_highs'] / breadth['new_lows']
                if breadth['new_lows'] > 0 else float('inf')
            )

        return breadth

    def _load_manifest(self) -> Dict:
        """Load data manifest from file."""
        if self.manifest_file.exists():
            with open(self.manifest_file, 'r') as f:
                return json.load(f)
        return {'symbols': {}, 'last_updated': None}

    def _update_manifest(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        record_count: int
    ):
        """Update data manifest with symbol information."""
        self.manifest['symbols'][symbol] = {
            'start_date': start_date,
            'end_date': end_date,
            'record_count': record_count,
            'last_updated': datetime.now().isoformat()
        }
        self.manifest['last_updated'] = datetime.now().isoformat()

        with open(self.manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)

    def _save_realtime_snapshot(
        self,
        symbol: str,
        data: pd.DataFrame,
        timestamp: str
    ):
        """Save realtime data snapshot."""
        filename = f"{symbol}_realtime_{timestamp}.csv"
        filepath = self.realtime_dir / filename
        data.to_csv(filepath)

    def get_sp100_symbols(self) -> List[str]:
        """
        Get S&P 100 constituent symbols.

        Returns:
            List of S&P 100 symbols
        """
        # S&P 100 symbols as of 2024
        sp100_symbols = [
            'AAPL', 'ABBV', 'ABT', 'ACN', 'ADBE', 'AIG', 'AMD', 'AMGN', 'AMT', 'AMZN',
            'AVGO', 'AXP', 'BA', 'BAC', 'BK', 'BKNG', 'BLK', 'BMY', 'BRK-B', 'C',
            'CAT', 'CHTR', 'CL', 'CMCSA', 'COF', 'COP', 'COST', 'CRM', 'CSCO', 'CVS',
            'CVX', 'DHR', 'DIS', 'DOW', 'DUK', 'EMR', 'EXC', 'F', 'FDX', 'GD',
            'GE', 'GILD', 'GM', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTC',
            'JNJ', 'JPM', 'KHC', 'KO', 'LIN', 'LLY', 'LMT', 'LOW', 'MA', 'MCD',
            'MDLZ', 'MDT', 'MET', 'META', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'NEE',
            'NFLX', 'NKE', 'NVDA', 'ORCL', 'OXY', 'PEP', 'PFE', 'PG', 'PM', 'QCOM',
            'RTX', 'SBUX', 'SCHW', 'SO', 'SPG', 'T', 'TGT', 'TMO', 'TSLA', 'TXN',
            'UNH', 'UNP', 'UPS', 'USB', 'V', 'VZ', 'WBA', 'WFC', 'WMT', 'XOM'
        ]

        return sp100_symbols


def main():
    """Example usage of SimpleDataCollector."""
    collector = SimpleDataCollector()

    # Example: Fetch SPY historical data
    print("Fetching SPY historical data...")
    spy_data = collector.fetch_historical("SPY", start_date="2019-01-01")

    if "SPY" in spy_data:
        print(f"SPY data shape: {spy_data['SPY'].shape}")
        print(f"SPY latest close: ${spy_data['SPY']['close'].iloc[-1]:.2f}")

        # Save to CSV
        collector.save_to_csv(spy_data)

    # Example: Fetch VIX data
    print("\nFetching VIX data...")
    vix_data = collector.get_vix_data()
    if not vix_data.empty:
        print(f"Latest VIX: {vix_data['close'].iloc[-1]:.2f}")
        print(f"VIX regime: {vix_data['vix_regime'].iloc[-1]}")

    # Example: Fetch realtime data
    print("\nFetching realtime data...")
    realtime = collector.fetch_realtime(["SPY", "QQQ"], period="1d", interval="15m")
    for symbol, data in realtime.items():
        print(f"{symbol} realtime records: {len(data)}")


if __name__ == "__main__":
    main()