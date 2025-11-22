#!/usr/bin/env python3
"""
Download Historical Data Script
================================
Download SPY, VIX, and S&P 100 stocks historical data (2019-2024).
Create data manifest and prepare for regime detection validation.

Author: Backend Systems Architect
Date: November 2024
Version: 1.0
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.umtrading.data.simple_data_collector import SimpleDataCollector
import pandas as pd
import logging
from datetime import datetime
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_all_data(
    start_date: str = "2019-01-01",
    end_date: str = None,
    batch_size: int = 10
):
    """
    Download all required historical data.

    Args:
        start_date: Start date for historical data
        end_date: End date (defaults to today)
        batch_size: Number of symbols to download in each batch
    """
    collector = SimpleDataCollector()

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    logger.info(f"Starting data download from {start_date} to {end_date}")

    # 1. Download SPY data (market index)
    logger.info("=" * 50)
    logger.info("Downloading SPY (S&P 500 ETF) data...")
    spy_data = collector.fetch_historical("SPY", start_date, end_date)
    if "SPY" in spy_data:
        collector.save_to_csv(spy_data)
        logger.info(f"SPY: {len(spy_data['SPY'])} days of data downloaded")
    else:
        logger.error("Failed to download SPY data")

    # 2. Download VIX data (volatility index)
    logger.info("=" * 50)
    logger.info("Downloading VIX (Volatility Index) data...")
    vix_data = collector.get_vix_data(start_date, end_date)
    if not vix_data.empty:
        collector.save_to_csv({"VIX": vix_data}, subfolder="historical")
        logger.info(f"VIX: {len(vix_data)} days of data downloaded")
    else:
        logger.error("Failed to download VIX data")

    # 3. Download S&P 100 stocks
    logger.info("=" * 50)
    logger.info("Downloading S&P 100 constituent stocks...")

    sp100_symbols = collector.get_sp100_symbols()
    total_symbols = len(sp100_symbols)
    downloaded = []
    failed = []

    # Download in batches to avoid overwhelming the API
    for i in range(0, total_symbols, batch_size):
        batch = sp100_symbols[i:i+batch_size]
        logger.info(f"Downloading batch {i//batch_size + 1}/{(total_symbols + batch_size - 1)//batch_size}: {batch}")

        batch_data = collector.fetch_historical(batch, start_date, end_date)

        # Save each successful download
        for symbol, data in batch_data.items():
            if not data.empty:
                collector.save_to_csv({symbol: data})
                downloaded.append(symbol)
                logger.info(f"  {symbol}: {len(data)} days downloaded")
            else:
                failed.append(symbol)
                logger.warning(f"  {symbol}: No data")

        # Small delay between batches (handled by yfinance internally)

    # 4. Create summary report
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("-" * 30)
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Total S&P 100 symbols: {total_symbols}")
    logger.info(f"Successfully downloaded: {len(downloaded)}")
    logger.info(f"Failed downloads: {len(failed)}")

    if failed:
        logger.warning(f"Failed symbols: {failed}")

    # 5. Create enhanced manifest
    manifest_path = collector.data_dir / "download_summary.json"
    manifest = {
        'download_date': datetime.now().isoformat(),
        'start_date': start_date,
        'end_date': end_date,
        'market_index': ['SPY'],
        'volatility_index': ['VIX'],
        'sp100_downloaded': downloaded,
        'sp100_failed': failed,
        'total_symbols': len(downloaded) + 2,  # +2 for SPY and VIX
        'statistics': {
            'success_rate': len(downloaded) / total_symbols * 100 if total_symbols > 0 else 0
        }
    }

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Download summary saved to {manifest_path}")

    return downloaded, failed


def verify_data_quality(data_dir: str = None):
    """
    Verify the quality of downloaded data.

    Args:
        data_dir: Directory containing the data
    """
    if data_dir is None:
        data_dir = Path("/Users/waiyang/Desktop/repo/dreamers-v2/data")
    else:
        data_dir = Path(data_dir)

    historical_dir = data_dir / "historical"

    logger.info("=" * 50)
    logger.info("DATA QUALITY VERIFICATION")
    logger.info("-" * 30)

    quality_report = {
        'total_files': 0,
        'symbols_checked': [],
        'issues': [],
        'statistics': {}
    }

    # Check all CSV files
    csv_files = list(historical_dir.glob("*.csv"))
    quality_report['total_files'] = len(csv_files)

    for csv_file in csv_files:
        symbol = csv_file.stem.split('_')[0]

        try:
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

            # Quality checks
            checks = {
                'rows': len(df),
                'missing_values': df.isnull().sum().sum(),
                'date_gaps': 0,
                'has_volume': 'volume' in df.columns,
                'has_sma': 'sma_50' in df.columns and 'sma_200' in df.columns
            }

            # Check for date gaps (weekends excluded)
            if len(df) > 1:
                date_diffs = pd.Series(df.index).diff()
                # Gaps > 3 days (to account for weekends)
                large_gaps = date_diffs[date_diffs > pd.Timedelta(days=3)]
                checks['date_gaps'] = len(large_gaps)

            quality_report['symbols_checked'].append(symbol)

            # Report issues
            if checks['missing_values'] > 0:
                quality_report['issues'].append(f"{symbol}: {checks['missing_values']} missing values")
            if checks['date_gaps'] > 5:
                quality_report['issues'].append(f"{symbol}: {checks['date_gaps']} date gaps")
            if checks['rows'] < 100:
                quality_report['issues'].append(f"{symbol}: Only {checks['rows']} rows")

            # Update statistics
            if symbol in ['SPY', 'VIX']:
                quality_report['statistics'][symbol] = checks

        except Exception as e:
            quality_report['issues'].append(f"{symbol}: Error reading file - {str(e)}")

    # Print report
    logger.info(f"Files checked: {quality_report['total_files']}")
    logger.info(f"Symbols verified: {len(quality_report['symbols_checked'])}")

    if quality_report['issues']:
        logger.warning(f"Issues found: {len(quality_report['issues'])}")
        for issue in quality_report['issues'][:10]:  # Show first 10 issues
            logger.warning(f"  - {issue}")
    else:
        logger.info("No data quality issues found!")

    # Save quality report
    report_path = data_dir / "data_quality_report.json"
    with open(report_path, 'w') as f:
        json.dump(quality_report, f, indent=2)

    logger.info(f"Quality report saved to {report_path}")

    return quality_report


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Download historical market data")
    parser.add_argument('--start', default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument('--end', default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument('--batch-size', type=int, default=10, help="Batch size for downloads")
    parser.add_argument('--verify-only', action='store_true', help="Only verify existing data")

    args = parser.parse_args()

    if args.verify_only:
        # Only run verification
        logger.info("Running data quality verification...")
        verify_data_quality()
    else:
        # Download data
        logger.info("Starting historical data download...")
        downloaded, failed = download_all_data(
            start_date=args.start,
            end_date=args.end,
            batch_size=args.batch_size
        )

        # Verify after download
        logger.info("\nVerifying downloaded data...")
        verify_data_quality()

        # Final summary
        logger.info("\n" + "=" * 50)
        logger.info("DOWNLOAD COMPLETE")
        logger.info(f"Successfully downloaded: {len(downloaded) + 2} symbols")  # +2 for SPY and VIX
        if failed:
            logger.info(f"Failed: {len(failed)} symbols")


if __name__ == "__main__":
    main()