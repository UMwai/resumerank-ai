"""
Scheduler for Patent Intelligence Pipeline

Provides scheduling capabilities for automated pipeline runs
and weekly email digests.
"""

import argparse
import signal
import sys
from datetime import datetime, time
from typing import Callable, Optional

import schedule
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .pipeline import PatentIntelligencePipeline
from .utils.config import get_config
from .utils.logger import get_logger, setup_logger

logger = get_logger(__name__)


class PatentIntelligenceScheduler:
    """
    Scheduler for automated patent intelligence pipeline runs.

    Supports:
    - Weekly full pipeline runs
    - Daily incremental updates
    - Configurable email digest timing
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the scheduler.

        Args:
            config_path: Path to configuration file.
        """
        self.config = get_config(config_path)
        self.pipeline: Optional[PatentIntelligencePipeline] = None
        self.scheduler: Optional[BlockingScheduler] = None

        # Setup logging
        setup_logger(
            log_level=self.config.get("logging.level", "INFO"),
            log_file=self.config.get("logging.log_file"),
        )

        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}. Shutting down...")
        if self.scheduler:
            self.scheduler.shutdown(wait=False)
        sys.exit(0)

    def _get_pipeline(self) -> PatentIntelligencePipeline:
        """Get or create pipeline instance."""
        if self.pipeline is None:
            self.pipeline = PatentIntelligencePipeline()
        return self.pipeline

    def run_weekly_pipeline(self):
        """
        Run the full weekly pipeline.

        This is the main scheduled task that:
        1. Extracts fresh data from all sources
        2. Calculates patent cliff scores
        3. Updates the database
        4. Exports calendar files
        5. Sends weekly email digest
        """
        logger.info("Starting scheduled weekly pipeline run...")

        try:
            pipeline = self._get_pipeline()

            results = pipeline.run_full_pipeline(
                top_n=self.config.top_drugs_count,
                load_db=True,
                send_email=self.config.get("email.enabled", False),
                export_files=True,
                output_dir="output",
            )

            if results["status"] == "SUCCESS":
                logger.info(
                    f"Weekly pipeline completed successfully. "
                    f"Events generated: {results.get('events_generated', 0)}"
                )
            else:
                logger.error(f"Weekly pipeline failed: {results.get('errors', [])}")

        except Exception as e:
            logger.error(f"Weekly pipeline error: {e}")

    def run_daily_update(self):
        """
        Run daily incremental update.

        Lighter-weight update that:
        1. Checks for new FDA approvals
        2. Updates existing records
        3. Recalculates scores for upcoming events
        """
        logger.info("Starting scheduled daily update...")

        try:
            pipeline = self._get_pipeline()

            # Run extraction only (no email)
            pipeline.extract_orange_book_data(top_n=self.config.top_drugs_count)
            pipeline.enrich_patent_data()
            pipeline.extract_anda_data()
            pipeline.generate_calendar()

            # Update database
            pipeline.load_to_database()

            logger.info("Daily update completed successfully")

        except Exception as e:
            logger.error(f"Daily update error: {e}")

    def run_email_digest(self):
        """Send the weekly email digest."""
        logger.info("Sending weekly email digest...")

        try:
            pipeline = self._get_pipeline()

            # Ensure we have current data
            if not pipeline.calendar_events:
                pipeline.extract_orange_book_data(top_n=self.config.top_drugs_count)
                pipeline.enrich_patent_data()
                pipeline.extract_anda_data()
                pipeline.generate_calendar()

            # Send email
            success = pipeline.send_weekly_digest()

            if success:
                logger.info("Weekly email digest sent successfully")
            else:
                logger.warning("Weekly email digest not sent (disabled or no recipients)")

        except Exception as e:
            logger.error(f"Email digest error: {e}")

    def start(
        self,
        weekly_day: str = "monday",
        weekly_hour: int = 8,
        daily_hour: int = 6,
        run_immediately: bool = False,
    ):
        """
        Start the scheduler.

        Args:
            weekly_day: Day of week for weekly pipeline (monday-sunday).
            weekly_hour: Hour for weekly pipeline (0-23).
            daily_hour: Hour for daily updates (0-23).
            run_immediately: Whether to run pipeline immediately on start.
        """
        logger.info("Starting Patent Intelligence Scheduler...")

        self.scheduler = BlockingScheduler()

        # Weekly full pipeline run
        self.scheduler.add_job(
            self.run_weekly_pipeline,
            CronTrigger(day_of_week=weekly_day, hour=weekly_hour, minute=0),
            id="weekly_pipeline",
            name="Weekly Patent Intelligence Pipeline",
            replace_existing=True,
        )
        logger.info(
            f"Scheduled weekly pipeline: {weekly_day} at {weekly_hour:02d}:00"
        )

        # Daily incremental update (skip on weekly day)
        days_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        weekly_day_num = days_map.get(weekly_day.lower(), 0)
        daily_days = ",".join(
            str(d) for d in range(7) if d != weekly_day_num
        )

        self.scheduler.add_job(
            self.run_daily_update,
            CronTrigger(day_of_week=daily_days, hour=daily_hour, minute=0),
            id="daily_update",
            name="Daily Patent Intelligence Update",
            replace_existing=True,
        )
        logger.info(f"Scheduled daily updates at {daily_hour:02d}:00")

        # Run immediately if requested
        if run_immediately:
            logger.info("Running initial pipeline...")
            self.run_weekly_pipeline()

        # Start scheduler
        logger.info("Scheduler started. Press Ctrl+C to exit.")

        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            logger.info("Scheduler stopped.")
        finally:
            if self.pipeline:
                self.pipeline.close()

    def run_once(self, send_email: bool = False):
        """
        Run the pipeline once and exit.

        Args:
            send_email: Whether to send email digest.
        """
        logger.info("Running pipeline once...")

        try:
            pipeline = self._get_pipeline()

            results = pipeline.run_full_pipeline(
                top_n=self.config.top_drugs_count,
                load_db=True,
                send_email=send_email,
                export_files=True,
                output_dir="output",
            )

            return results

        finally:
            if self.pipeline:
                self.pipeline.close()


def main():
    """Command-line interface for the scheduler."""
    parser = argparse.ArgumentParser(
        description="Patent Intelligence Pipeline Scheduler"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )

    parser.add_argument(
        "--mode",
        choices=["schedule", "once", "email"],
        default="once",
        help="Run mode: schedule (continuous), once (single run), email (send digest)",
    )

    parser.add_argument(
        "--weekly-day",
        type=str,
        default="monday",
        help="Day of week for weekly pipeline (default: monday)",
    )

    parser.add_argument(
        "--weekly-hour",
        type=int,
        default=8,
        help="Hour for weekly pipeline, 0-23 (default: 8)",
    )

    parser.add_argument(
        "--daily-hour",
        type=int,
        default=6,
        help="Hour for daily updates, 0-23 (default: 6)",
    )

    parser.add_argument(
        "--run-now",
        action="store_true",
        help="Run pipeline immediately on scheduler start",
    )

    parser.add_argument(
        "--send-email",
        action="store_true",
        help="Send email digest (for 'once' mode)",
    )

    args = parser.parse_args()

    scheduler = PatentIntelligenceScheduler(config_path=args.config)

    if args.mode == "schedule":
        scheduler.start(
            weekly_day=args.weekly_day,
            weekly_hour=args.weekly_hour,
            daily_hour=args.daily_hour,
            run_immediately=args.run_now,
        )

    elif args.mode == "once":
        results = scheduler.run_once(send_email=args.send_email)

        if results["status"] == "SUCCESS":
            print("\nPipeline completed successfully!")
            print(f"Events generated: {results.get('events_generated', 0)}")
        else:
            print("\nPipeline failed!")
            for error in results.get("errors", []):
                print(f"  - {error}")
            sys.exit(1)

    elif args.mode == "email":
        scheduler.run_email_digest()
        print("Email digest sent.")


if __name__ == "__main__":
    main()
