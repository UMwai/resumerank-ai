#!/usr/bin/env python3
"""
Validate Regime Detector Script
================================
Validate regime detection accuracy on known historical periods:
- 2020 COVID crash → Should detect BEAR
- 2021-2023 bull run → Should detect BULL
- 2022 correction → Should detect BEAR
Calculate accuracy percentage and generate validation report.

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
from src.umtrading.regime.simple_detector import (
    SimpleRegimeDetector,
    SimpleRegime,
    create_regime_report
)
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# Known historical regime periods for validation
KNOWN_PERIODS = {
    'bear': [
        ('2020-02-20', '2020-03-23'),  # COVID crash
        ('2022-01-03', '2022-06-16'),  # 2022 H1 bear market
        ('2022-08-16', '2022-10-12'),  # 2022 Q3 correction
    ],
    'bull': [
        ('2019-01-01', '2020-02-19'),  # Pre-COVID bull
        ('2020-03-24', '2021-12-31'),  # Post-COVID recovery bull
        ('2023-01-01', '2023-07-31'),  # 2023 H1 rally
        ('2023-10-27', '2024-03-31'),  # Late 2023 - Early 2024 rally
    ],
    'neutral': [
        ('2022-06-17', '2022-08-15'),  # Summer 2022 consolidation
        ('2023-08-01', '2023-10-26'),  # Late summer 2023 consolidation
    ]
}


def load_and_prepare_data(data_dir: str = None) -> tuple:
    """
    Load SPY and VIX data from saved CSV files.

    Args:
        data_dir: Directory containing the data

    Returns:
        Tuple of (spy_data, vix_data)
    """
    if data_dir is None:
        data_dir = Path("/Users/waiyang/Desktop/repo/dreamers-v2/data")
    else:
        data_dir = Path(data_dir)

    historical_dir = data_dir / "historical"

    logger.info("Loading historical data...")

    # Find and load SPY data
    spy_files = list(historical_dir.glob("SPY_*.csv"))
    if not spy_files:
        raise FileNotFoundError("No SPY data files found. Please run download_historical.py first.")

    spy_file = max(spy_files, key=lambda f: f.stat().st_mtime)  # Most recent
    spy_data = pd.read_csv(spy_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded SPY data: {len(spy_data)} days from {spy_file.name}")

    # Find and load VIX data
    vix_files = list(historical_dir.glob("VIX_*.csv")) + list(historical_dir.glob("^VIX_*.csv"))
    if not vix_files:
        raise FileNotFoundError("No VIX data files found. Please run download_historical.py first.")

    vix_file = max(vix_files, key=lambda f: f.stat().st_mtime)  # Most recent
    vix_data = pd.read_csv(vix_file, index_col=0, parse_dates=True)
    logger.info(f"Loaded VIX data: {len(vix_data)} days from {vix_file.name}")

    # Ensure we have required columns
    if 'sma_50' not in spy_data.columns or 'sma_200' not in spy_data.columns:
        logger.info("Calculating missing SMAs...")
        spy_data['sma_50'] = spy_data['close'].rolling(window=50).mean()
        spy_data['sma_200'] = spy_data['close'].rolling(window=200).mean()

    if 'vix_ma20' not in vix_data.columns:
        logger.info("Calculating VIX MA20...")
        vix_data['vix_ma20'] = vix_data['close'].rolling(window=20).mean()

    return spy_data, vix_data


def run_validation(spy_data: pd.DataFrame, vix_data: pd.DataFrame) -> dict:
    """
    Run regime detection validation on historical data.

    Args:
        spy_data: SPY price data with SMAs
        vix_data: VIX data

    Returns:
        Validation results dictionary
    """
    detector = SimpleRegimeDetector()

    logger.info("=" * 60)
    logger.info("RUNNING REGIME DETECTION VALIDATION")
    logger.info("=" * 60)

    # Run detection for entire period
    logger.info("\nDetecting regimes for full historical period...")

    all_regimes = []
    for date in spy_data.index:
        if date not in vix_data.index:
            continue

        # Need at least 200 days of history for SMA200
        if len(spy_data[:date]) < 200:
            continue

        regime, confidence, indicators = detector.detect_regime(
            spy_data[:date],
            vix_data[:date],
            date
        )

        all_regimes.append({
            'date': date,
            'regime': regime.value,
            'confidence': confidence,
            'spy_close': spy_data.loc[date, 'close'],
            'vix_close': vix_data.loc[date, 'close'],
            'indicators': indicators
        })

    # Create DataFrame of all regimes
    regime_df = pd.DataFrame(all_regimes)
    regime_df.set_index('date', inplace=True)

    # Validate against known periods
    logger.info("\nValidating against known regime periods...")
    validation_results = detector.validate_on_historical(
        spy_data,
        vix_data,
        KNOWN_PERIODS
    )

    # Add regime DataFrame to results
    validation_results['regime_history'] = regime_df

    return validation_results


def create_visualizations(
    spy_data: pd.DataFrame,
    vix_data: pd.DataFrame,
    validation_results: dict,
    save_dir: str = None
):
    """
    Create visualization plots for regime detection validation.

    Args:
        spy_data: SPY price data
        vix_data: VIX data
        validation_results: Validation results
        save_dir: Directory to save plots
    """
    if save_dir is None:
        save_dir = Path("/Users/waiyang/Desktop/repo/dreamers-v2/data/plots")
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    regime_df = validation_results.get('regime_history')
    if regime_df is None:
        logger.warning("No regime history to visualize")
        return

    # Filter to 2019 onwards for cleaner plots
    regime_df = regime_df[regime_df.index >= '2019-01-01']
    spy_data = spy_data[spy_data.index >= '2019-01-01']
    vix_data = vix_data[vix_data.index >= '2019-01-01']

    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

    # 1. SPY Price with Regime Background
    ax1 = axes[0]

    # Plot SPY price
    ax1.plot(spy_data.index, spy_data['close'], label='SPY Close', color='black', linewidth=1)
    ax1.plot(spy_data.index, spy_data['sma_50'], label='SMA 50', color='blue', alpha=0.7, linewidth=0.8)
    ax1.plot(spy_data.index, spy_data['sma_200'], label='SMA 200', color='red', alpha=0.7, linewidth=0.8)

    # Add regime backgrounds
    regime_colors = {'bull': 'green', 'bear': 'red', 'neutral': 'yellow'}
    for regime in ['bull', 'bear', 'neutral']:
        regime_mask = regime_df['regime'] == regime
        if regime_mask.any():
            ax1.fill_between(
                regime_df.index,
                spy_data['close'].min() * 0.95,
                spy_data['close'].max() * 1.05,
                where=regime_mask,
                alpha=0.2,
                color=regime_colors[regime],
                label=f'{regime.upper()} regime'
            )

    ax1.set_ylabel('SPY Price ($)')
    ax1.set_title('SPY Price with Detected Market Regimes')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. VIX Level
    ax2 = axes[1]
    ax2.plot(vix_data.index, vix_data['close'], label='VIX', color='purple', linewidth=1)
    ax2.axhline(y=20, color='orange', linestyle='--', alpha=0.5, label='Low/Medium threshold (20)')
    ax2.axhline(y=25, color='red', linestyle='--', alpha=0.5, label='Medium/High threshold (25)')
    ax2.fill_between(vix_data.index, 0, vix_data['close'], alpha=0.3, color='purple')
    ax2.set_ylabel('VIX Level')
    ax2.set_title('VIX Volatility Index')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Regime Confidence
    ax3 = axes[2]
    ax3.plot(regime_df.index, regime_df['confidence'] * 100, label='Detection Confidence', color='darkblue', linewidth=1)
    ax3.axhline(y=70, color='green', linestyle='--', alpha=0.5, label='High confidence (70%)')
    ax3.fill_between(regime_df.index, 0, regime_df['confidence'] * 100, alpha=0.3, color='darkblue')
    ax3.set_ylabel('Confidence (%)')
    ax3.set_title('Regime Detection Confidence')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.set_ylim([0, 100])
    ax3.grid(True, alpha=0.3)

    # 4. Known Periods Validation
    ax4 = axes[3]

    # Create validation accuracy over time
    validation_accuracy = []
    for date in regime_df.index:
        # Find which known period this date falls into
        true_regime = None
        for regime, periods in KNOWN_PERIODS.items():
            for start_str, end_str in periods:
                start = pd.to_datetime(start_str)
                end = pd.to_datetime(end_str)
                if start <= date <= end:
                    true_regime = regime
                    break
            if true_regime:
                break

        if true_regime:
            detected_regime = regime_df.loc[date, 'regime']
            is_correct = 1 if detected_regime == true_regime else 0
            validation_accuracy.append({
                'date': date,
                'correct': is_correct,
                'true_regime': true_regime,
                'detected_regime': detected_regime
            })

    if validation_accuracy:
        val_df = pd.DataFrame(validation_accuracy)
        val_df.set_index('date', inplace=True)

        # Calculate rolling accuracy
        val_df['rolling_accuracy'] = val_df['correct'].rolling(window=20, min_periods=1).mean() * 100

        ax4.plot(val_df.index, val_df['rolling_accuracy'], label='20-day Rolling Accuracy', color='green', linewidth=1.5)
        ax4.axhline(y=validation_results['total_accuracy'] * 100, color='blue', linestyle='--',
                   label=f"Overall Accuracy: {validation_results['total_accuracy']:.1%}")
        ax4.fill_between(val_df.index, 0, val_df['rolling_accuracy'], alpha=0.3, color='green')

    ax4.set_ylabel('Accuracy (%)')
    ax4.set_xlabel('Date')
    ax4.set_title('Validation Accuracy on Known Periods')
    ax4.legend(loc='lower right', fontsize=9)
    ax4.set_ylim([0, 100])
    ax4.grid(True, alpha=0.3)

    # Adjust layout and save
    plt.tight_layout()
    plot_path = save_dir / "regime_validation_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {plot_path}")
    plt.close()

    # Create confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(8, 6))

    cm = validation_results['confusion_matrix']
    cm_data = []
    for actual in ['bull', 'bear', 'neutral']:
        row = [cm[actual]['bull'], cm[actual]['bear'], cm[actual]['neutral']]
        cm_data.append(row)

    cm_df = pd.DataFrame(cm_data,
                        index=['Bull (Actual)', 'Bear (Actual)', 'Neutral (Actual)'],
                        columns=['Bull (Pred)', 'Bear (Pred)', 'Neutral (Pred)'])

    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix: Actual vs Predicted Regimes')
    plt.tight_layout()

    cm_path = save_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved confusion matrix to {cm_path}")
    plt.close()


def save_validation_results(validation_results: dict, output_dir: str = None):
    """
    Save validation results to files.

    Args:
        validation_results: Validation results dictionary
        output_dir: Directory to save results
    """
    if output_dir is None:
        output_dir = Path("/Users/waiyang/Desktop/repo/dreamers-v2/data/validation")
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save regime history to CSV
    if 'regime_history' in validation_results:
        regime_path = output_dir / "regime_history.csv"
        validation_results['regime_history'].to_csv(regime_path)
        logger.info(f"Saved regime history to {regime_path}")

    # Save validation metrics (exclude DataFrame)
    metrics_dict = {k: v for k, v in validation_results.items() if k != 'regime_history'}

    # Convert period_results details to summary (too large otherwise)
    if 'period_results' in metrics_dict:
        for period in metrics_dict['period_results']:
            if 'details' in period:
                # Just keep summary stats instead of all daily details
                details = period['details']
                period['details_summary'] = {
                    'total_days': len(details),
                    'correct_days': sum(1 for d in details if d['correct']),
                    'avg_confidence': np.mean([d['confidence'] for d in details])
                }
                del period['details']

    metrics_path = output_dir / "validation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=str)
    logger.info(f"Saved validation metrics to {metrics_path}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate regime detection on historical data")
    parser.add_argument('--data-dir', help="Directory containing historical data")
    parser.add_argument('--output-dir', help="Directory to save results")
    parser.add_argument('--no-plots', action='store_true', help="Skip creating plots")

    args = parser.parse_args()

    try:
        # Load data
        spy_data, vix_data = load_and_prepare_data(args.data_dir)

        # Run validation
        validation_results = run_validation(spy_data, vix_data)

        # Create regime detector for report
        detector = SimpleRegimeDetector()
        detector.regime_history = validation_results.get('period_results', [])

        # Generate and print report
        report = create_regime_report(detector, validation_results)
        print("\n" + report)

        # Save results
        save_validation_results(validation_results, args.output_dir)

        # Create visualizations
        if not args.no_plots:
            logger.info("\nCreating visualizations...")
            create_visualizations(spy_data, vix_data, validation_results)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Overall Accuracy: {validation_results['total_accuracy']:.1%}")

        # Check specific known periods
        logger.info("\nKey Period Validation:")
        key_periods = [
            ("2020 COVID Crash", "2020-02-20 to 2020-03-23", "bear"),
            ("2021 Bull Run", "2020-03-24 to 2021-12-31", "bull"),
            ("2022 Bear Market", "2022-01-03 to 2022-06-16", "bear"),
            ("2023 H1 Rally", "2023-01-01 to 2023-07-31", "bull"),
        ]

        for name, period_str, expected in key_periods:
            # Find matching period in results
            for period in validation_results.get('period_results', []):
                if expected in period['period'] or period['true_regime'] == expected:
                    if period_str in period['period'] or any(date in period['period'] for date in period_str.split(' to ')):
                        acc = period['accuracy']
                        status = "✓" if acc > 0.7 else "✗"
                        logger.info(f"  {status} {name}: {acc:.1%} accuracy")
                        break

    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please run download_historical.py first to download the required data.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()