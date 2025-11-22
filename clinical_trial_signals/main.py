#!/usr/bin/env python3
"""
Clinical Trial Signal Detection System - Main Orchestration Script

This script coordinates all components of the system:
1. Fetches trial data from ClinicalTrials.gov
2. Fetches SEC 8-K filings
3. Detects changes in trial data
4. Calculates composite scores
5. Sends daily email digest

Usage:
    python main.py --full          # Run full pipeline (fetch + detect + score + email)
    python main.py --fetch         # Only fetch new data
    python main.py --detect        # Only run change detection
    python main.py --score         # Only calculate scores
    python main.py --email         # Only send email digest
    python main.py --init-db       # Initialize database schema
    python main.py --dry-run       # Run without database writes or emails
"""
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

from config import config, validate_config
from database.connection import get_db_connection, MockDatabaseConnection
from database.models import Trial, Company, TrialSignal
from scrapers.clinicaltrials import ClinicalTrialsScraper, BIOTECH_SPONSOR_MAPPING
from scrapers.sec_edgar import SECEdgarScraper
from utils.change_detection import ChangeDetector
from scoring.signal_scorer import SignalScorer
from alerts.email_digest import EmailDigest


# Configure logging
def setup_logging(log_level: str = None, log_file: str = None):
    """Configure logging for the application."""
    level = getattr(logging, (log_level or config.log_level).upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file or config.log_file:
        handlers.append(logging.FileHandler(log_file or config.log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )


logger = logging.getLogger(__name__)


class ClinicalTrialPipeline:
    """
    Main pipeline orchestrator for Clinical Trial Signal Detection.
    """

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run or config.dry_run
        self.db = MockDatabaseConnection() if self.dry_run else get_db_connection()

        self.trial_scraper = ClinicalTrialsScraper()
        self.sec_scraper = SECEdgarScraper()
        self.change_detector = ChangeDetector()
        self.scorer = SignalScorer()
        self.email_digest = EmailDigest()

    def initialize_database(self) -> bool:
        """Initialize database schema."""
        logger.info("Initializing database schema...")
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would initialize database schema")
                return True

            db = get_db_connection()
            db.init_schema()
            logger.info("Database schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    def fetch_trials(self, limit: int = 20) -> dict:
        """
        Fetch clinical trial data from ClinicalTrials.gov.

        Returns:
            Dictionary with fetch statistics
        """
        logger.info(f"Fetching top {limit} Phase 3 trials from ClinicalTrials.gov...")

        stats = {
            "new_trials": 0,
            "updated_trials": 0,
            "errors": []
        }

        try:
            # Build sponsor-to-ticker mapping from database
            company_mapping = {}
            if not self.dry_run:
                companies = Company.get_all()
                for c in companies:
                    company_mapping[c.company_name] = c.ticker

            # Add known mappings
            company_mapping.update(BIOTECH_SPONSOR_MAPPING)

            # Fetch and store trials
            new_count, updated_count = self.trial_scraper.fetch_and_store_trials(
                conditions=config.monitoring.priority_indications,
                limit=limit,
                company_tickers=company_mapping
            )

            stats["new_trials"] = new_count
            stats["updated_trials"] = updated_count

            logger.info(f"Fetched {new_count} new trials, updated {updated_count} existing")

        except Exception as e:
            logger.error(f"Error fetching trials: {e}")
            stats["errors"].append(str(e))

        return stats

    def fetch_sec_filings(self, days_back: int = 7) -> dict:
        """
        Fetch SEC 8-K filings for monitored companies.

        Returns:
            Dictionary with fetch statistics
        """
        logger.info(f"Fetching SEC 8-K filings from last {days_back} days...")

        stats = {
            "filings_processed": 0,
            "signals_found": 0,
            "errors": []
        }

        try:
            # Get company tickers from database
            if self.dry_run:
                tickers = ["MRNA", "NVAX", "IONS"]  # Test tickers
            else:
                companies = Company.get_all()
                tickers = [c.ticker for c in companies]

            # Fetch filings and analyze
            results = self.sec_scraper.fetch_filings_for_companies(
                tickers=tickers,
                days_back=days_back
            )

            stats["filings_processed"] = len(results)

            # Create signals from SEC filings
            for ticker, filing, signals in results:
                for signal_data in signals:
                    # Find associated trial (if any)
                    trial_id = self._find_trial_for_company(ticker)

                    if trial_id and not self.dry_run:
                        signal = TrialSignal(
                            trial_id=trial_id,
                            signal_type=signal_data["signal_type"],
                            signal_value=signal_data["description"],
                            signal_weight=config.signal.weights.get(
                                signal_data["signal_type"], 0
                            ),
                            source="sec_edgar",
                            source_url=filing.filing_url,
                            raw_data=signal_data
                        )
                        signal.save()
                        stats["signals_found"] += 1

            logger.info(
                f"Processed {stats['filings_processed']} filings, "
                f"found {stats['signals_found']} signals"
            )

        except Exception as e:
            logger.error(f"Error fetching SEC filings: {e}")
            stats["errors"].append(str(e))

        return stats

    def _find_trial_for_company(self, ticker: str) -> str:
        """Find a trial associated with a company ticker."""
        if self.dry_run:
            return None

        trials = Trial.get_all()
        for trial in trials:
            if trial.company_ticker == ticker:
                return trial.trial_id
        return None

    def run_change_detection(self) -> dict:
        """
        Run change detection on all monitored trials.

        Returns:
            Dictionary with detection statistics
        """
        logger.info("Running change detection on monitored trials...")

        stats = {
            "trials_checked": 0,
            "changes_detected": 0,
            "signals_created": 0,
            "errors": []
        }

        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would run change detection")
                return stats

            trials = Trial.get_monitored()
            stats["trials_checked"] = len(trials)

            for trial in trials:
                try:
                    signals = self.change_detector.process_trial_changes(trial)
                    if signals:
                        stats["changes_detected"] += 1
                        stats["signals_created"] += len(signals)
                except Exception as e:
                    logger.warning(f"Error processing {trial.trial_id}: {e}")
                    stats["errors"].append(f"{trial.trial_id}: {str(e)}")

            logger.info(
                f"Checked {stats['trials_checked']} trials, "
                f"detected {stats['changes_detected']} with changes, "
                f"created {stats['signals_created']} signals"
            )

        except Exception as e:
            logger.error(f"Error in change detection: {e}")
            stats["errors"].append(str(e))

        return stats

    def calculate_scores(self) -> dict:
        """
        Calculate composite scores for all trials.

        Returns:
            Dictionary with scoring statistics
        """
        logger.info("Calculating composite scores...")

        stats = {
            "trials_scored": 0,
            "strong_buys": 0,
            "shorts": 0,
            "errors": []
        }

        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would calculate scores")
                return stats

            scores = self.scorer.score_all_trials(lookback_days=30)
            stats["trials_scored"] = len(scores)

            for score in scores:
                if score.recommendation in ("STRONG_BUY", "BUY"):
                    stats["strong_buys"] += 1
                elif score.recommendation in ("SHORT", "STRONG_SHORT"):
                    stats["shorts"] += 1

            logger.info(
                f"Scored {stats['trials_scored']} trials: "
                f"{stats['strong_buys']} buys, {stats['shorts']} shorts"
            )

        except Exception as e:
            logger.error(f"Error calculating scores: {e}")
            stats["errors"].append(str(e))

        return stats

    def send_email_digest(self) -> dict:
        """
        Generate and send daily email digest.

        Returns:
            Dictionary with email statistics
        """
        logger.info("Generating email digest...")

        stats = {
            "sent": False,
            "recipients": 0,
            "signals_included": 0,
            "errors": []
        }

        try:
            digest = self.email_digest.generate_digest(days=1)
            stats["signals_included"] = digest.signals_count

            if self.dry_run:
                logger.info("[DRY RUN] Would send email digest")
                self.email_digest.print_digest(days=1)
                return stats

            if config.email.enabled:
                success = self.email_digest.send_digest(digest)
                stats["sent"] = success
                stats["recipients"] = len(config.email.to_emails)
            else:
                logger.info("Email sending disabled, printing digest instead")
                self.email_digest.print_digest(days=1)

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            stats["errors"].append(str(e))

        return stats

    def run_full_pipeline(self) -> dict:
        """
        Run the complete pipeline.

        Returns:
            Dictionary with all statistics
        """
        logger.info("=" * 60)
        logger.info("Starting Clinical Trial Signal Detection Pipeline")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Dry run: {self.dry_run}")
        logger.info("=" * 60)

        results = {
            "start_time": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "stages": {}
        }

        # Stage 1: Fetch trials
        results["stages"]["fetch_trials"] = self.fetch_trials(limit=20)

        # Stage 2: Fetch SEC filings
        results["stages"]["fetch_sec"] = self.fetch_sec_filings(days_back=7)

        # Stage 3: Change detection
        results["stages"]["change_detection"] = self.run_change_detection()

        # Stage 4: Score calculation
        results["stages"]["scoring"] = self.calculate_scores()

        # Stage 5: Email digest
        results["stages"]["email"] = self.send_email_digest()

        results["end_time"] = datetime.now().isoformat()

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: dict) -> None:
        """Print pipeline execution summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 60)

        for stage, stats in results.get("stages", {}).items():
            logger.info(f"\n{stage.upper()}:")
            for key, value in stats.items():
                if key != "errors":
                    logger.info(f"  {key}: {value}")
            if stats.get("errors"):
                for error in stats["errors"]:
                    logger.warning(f"  ERROR: {error}")

        logger.info("")
        logger.info(f"Completed at: {results.get('end_time', 'N/A')}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Clinical Trial Signal Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode arguments (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--full", action="store_true",
        help="Run full pipeline (default)"
    )
    mode_group.add_argument(
        "--fetch", action="store_true",
        help="Only fetch new data from sources"
    )
    mode_group.add_argument(
        "--detect", action="store_true",
        help="Only run change detection"
    )
    mode_group.add_argument(
        "--score", action="store_true",
        help="Only calculate scores"
    )
    mode_group.add_argument(
        "--email", action="store_true",
        help="Only send email digest"
    )
    mode_group.add_argument(
        "--init-db", action="store_true",
        help="Initialize database schema"
    )

    # Options
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without database writes or sending emails"
    )
    parser.add_argument(
        "--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO", help="Logging level"
    )
    parser.add_argument(
        "--trials", type=int, default=20,
        help="Number of trials to fetch (default: 20)"
    )
    parser.add_argument(
        "--days", type=int, default=7,
        help="Days to look back for filings/signals (default: 7)"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level)

    # Validate configuration
    issues = validate_config(config)
    if issues and not args.dry_run:
        logger.warning("Configuration issues detected:")
        for issue in issues:
            logger.warning(f"  - {issue}")

    # Create pipeline
    pipeline = ClinicalTrialPipeline(dry_run=args.dry_run)

    # Execute based on mode
    if args.init_db:
        success = pipeline.initialize_database()
        sys.exit(0 if success else 1)

    elif args.fetch:
        stats = pipeline.fetch_trials(limit=args.trials)
        stats.update(pipeline.fetch_sec_filings(days_back=args.days))

    elif args.detect:
        stats = pipeline.run_change_detection()

    elif args.score:
        stats = pipeline.calculate_scores()

    elif args.email:
        stats = pipeline.send_email_digest()

    else:
        # Default: full pipeline
        stats = pipeline.run_full_pipeline()

    # Exit with error if there were errors
    if isinstance(stats, dict):
        all_errors = []
        if "errors" in stats:
            all_errors.extend(stats["errors"])
        if "stages" in stats:
            for stage_stats in stats["stages"].values():
                all_errors.extend(stage_stats.get("errors", []))

        if all_errors:
            logger.error(f"Pipeline completed with {len(all_errors)} error(s)")
            sys.exit(1)

    logger.info("Pipeline completed successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
