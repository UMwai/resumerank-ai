#!/usr/bin/env python3
"""
UMTrading Main Execution Script
================================
Main script to run the simplified regime-adaptive trading platform.
Combines data collection, regime detection, and validation.

Author: Backend Systems Architect
Date: November 2024
Version: 1.0

Usage:
    python main.py download     # Download historical data
    python main.py validate     # Validate regime detection
    python main.py realtime     # Run real-time monitoring
    python main.py full         # Run complete pipeline
"""

import sys
import os
from pathlib import Path
import logging
from datetime import datetime, timedelta
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.umtrading.data.simple_data_collector import SimpleDataCollector
from src.umtrading.regime.simple_detector import SimpleRegimeDetector, SimpleRegime
from src.umtrading.data import download_historical
from src.umtrading.regime import validate_detector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_data(start_date: str = "2019-01-01", end_date: str = None):
    """
    Download historical data for SPY, VIX, and S&P 100 stocks.

    Args:
        start_date: Start date for historical data
        end_date: End date (defaults to today)
    """
    logger.info("=" * 70)
    logger.info("STARTING DATA DOWNLOAD")
    logger.info("=" * 70)

    # Use download_historical module
    downloaded, failed = download_historical.download_all_data(
        start_date=start_date,
        end_date=end_date,
        batch_size=10
    )

    # Verify data quality
    logger.info("\nVerifying data quality...")
    download_historical.verify_data_quality()

    logger.info(f"\nDownload complete. Downloaded {len(downloaded) + 2} symbols.")
    if failed:
        logger.warning(f"Failed to download {len(failed)} symbols: {failed[:5]}...")

    return downloaded, failed


def validate_regimes():
    """
    Validate regime detection on historical data with known periods.
    """
    logger.info("=" * 70)
    logger.info("STARTING REGIME VALIDATION")
    logger.info("=" * 70)

    try:
        # Load data
        spy_data, vix_data = validate_detector.load_and_prepare_data()

        # Run validation
        validation_results = validate_detector.run_validation(spy_data, vix_data)

        # Create detector for report
        detector = SimpleRegimeDetector()

        # Generate report
        report = validate_detector.create_regime_report(detector, validation_results)
        print("\n" + report)

        # Save results
        validate_detector.save_validation_results(validation_results)

        # Create visualizations
        logger.info("\nCreating visualizations...")
        validate_detector.create_visualizations(spy_data, vix_data, validation_results)

        logger.info(f"\nValidation complete. Overall accuracy: {validation_results['total_accuracy']:.1%}")

        return validation_results

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return None


def run_realtime_monitoring(symbols: list = None, update_interval: int = 900):
    """
    Run real-time regime monitoring (15-minute updates).

    Args:
        symbols: List of symbols to monitor (defaults to SPY, QQQ, VIX)
        update_interval: Update interval in seconds (default 900 = 15 minutes)
    """
    logger.info("=" * 70)
    logger.info("STARTING REAL-TIME REGIME MONITORING")
    logger.info("=" * 70)

    if symbols is None:
        symbols = ["SPY", "QQQ", "^VIX"]

    collector = SimpleDataCollector()
    detector = SimpleRegimeDetector()

    logger.info(f"Monitoring symbols: {symbols}")
    logger.info(f"Update interval: {update_interval} seconds")
    logger.info("Press Ctrl+C to stop monitoring\n")

    try:
        import time

        while True:
            timestamp = datetime.now()
            logger.info(f"\n[{timestamp.strftime('%Y-%m-%d %H:%M:%S')}] Fetching real-time data...")

            # Fetch real-time data
            realtime_data = collector.fetch_realtime(symbols, period="5d", interval="15m")

            if "SPY" in realtime_data and "^VIX" in realtime_data:
                spy_data = realtime_data["SPY"]
                vix_data = realtime_data["^VIX"]

                # Calculate SMAs if not present
                if 'sma_50' not in spy_data.columns:
                    spy_data['sma_50'] = spy_data['close'].rolling(window=50).mean()
                    spy_data['sma_200'] = spy_data['close'].rolling(window=200).mean()

                if 'vix_ma20' not in vix_data.columns:
                    vix_data['vix_ma20'] = vix_data['close'].rolling(window=20).mean()

                # Detect current regime
                regime, confidence, indicators = detector.detect_regime(
                    spy_data,
                    vix_data
                )

                # Display current status
                logger.info(f"Current Market Status:")
                logger.info(f"  SPY: ${spy_data['close'].iloc[-1]:.2f}")
                logger.info(f"  VIX: {vix_data['close'].iloc[-1]:.2f}")
                logger.info(f"  Regime: {regime.value.upper()}")
                logger.info(f"  Confidence: {confidence:.1%}")

                # Generate trading signals based on regime
                if regime == SimpleRegime.BULL:
                    logger.info("  Signal: RISK-ON - Consider increasing equity allocation")
                elif regime == SimpleRegime.BEAR:
                    logger.info("  Signal: RISK-OFF - Consider defensive positioning")
                else:
                    logger.info("  Signal: NEUTRAL - Maintain balanced allocation")

                # Save snapshot
                collector.save_to_csv(realtime_data, subfolder="realtime")

            else:
                logger.warning("Failed to fetch required data")

            # Wait for next update
            logger.info(f"\nNext update in {update_interval} seconds...")
            time.sleep(update_interval)

    except KeyboardInterrupt:
        logger.info("\n\nMonitoring stopped by user.")
    except Exception as e:
        logger.error(f"Error in real-time monitoring: {str(e)}")
        import traceback
        traceback.print_exc()


def run_full_pipeline():
    """
    Run the complete pipeline: download, validate, and monitor.
    """
    logger.info("=" * 70)
    logger.info("RUNNING COMPLETE PIPELINE")
    logger.info("=" * 70)

    # Step 1: Download data
    logger.info("\nStep 1: Downloading historical data...")
    downloaded, failed = download_data()

    if not downloaded and not failed:
        logger.error("No data downloaded. Exiting.")
        return

    # Step 2: Validate regime detection
    logger.info("\nStep 2: Validating regime detection...")
    validation_results = validate_regimes()

    if validation_results is None:
        logger.error("Validation failed. Exiting.")
        return

    # Display summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE - SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Data Downloaded: {len(downloaded) + 2} symbols")
    logger.info(f"Validation Accuracy: {validation_results['total_accuracy']:.1%}")

    # Display regime-specific performance
    logger.info("\nRegime Detection Performance:")
    for regime, metrics in validation_results['metrics'].items():
        logger.info(f"  {regime.upper()}:")
        logger.info(f"    Precision: {metrics['precision']:.1%}")
        logger.info(f"    Recall: {metrics['recall']:.1%}")
        logger.info(f"    F1 Score: {metrics['f1_score']:.1%}")

    # Step 3: Offer to start real-time monitoring
    logger.info("\n" + "=" * 70)
    response = input("\nStart real-time monitoring? (y/n): ")
    if response.lower() == 'y':
        run_realtime_monitoring()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="UMTrading - Simple Regime-Adaptive Trading Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download              # Download historical data
  python main.py validate              # Validate regime detection
  python main.py realtime              # Start real-time monitoring
  python main.py full                  # Run complete pipeline
  python main.py download --start 2020-01-01  # Download from specific date
        """
    )

    parser.add_argument(
        'command',
        choices=['download', 'validate', 'realtime', 'full'],
        help='Command to execute'
    )

    parser.add_argument(
        '--start',
        default="2019-01-01",
        help='Start date for data download (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        default=None,
        help='End date for data download (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Symbols to monitor in real-time mode'
    )

    parser.add_argument(
        '--interval',
        type=int,
        default=900,
        help='Update interval in seconds for real-time mode (default: 900)'
    )

    args = parser.parse_args()

    # Print header
    print("\n" + "=" * 70)
    print(" UMTrading - Simple Regime-Adaptive Trading Platform")
    print(" Version 1.0 - November 2024")
    print("=" * 70 + "\n")

    # Execute command
    if args.command == 'download':
        download_data(start_date=args.start, end_date=args.end)

    elif args.command == 'validate':
        validation_results = validate_regimes()
        if validation_results:
            logger.info("\nValidation successful!")

    elif args.command == 'realtime':
        run_realtime_monitoring(symbols=args.symbols, update_interval=args.interval)

    elif args.command == 'full':
        run_full_pipeline()


if __name__ == "__main__":
    main()