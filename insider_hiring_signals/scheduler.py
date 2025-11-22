#!/usr/bin/env python3
"""
Scheduler for Insider Activity + Hiring Signals System

Runs scrapers and scoring on a schedule using APScheduler.
Can run as a daemon or in foreground.

Usage:
    python scheduler.py                 # Run in foreground
    python scheduler.py --daemon        # Run as daemon (background)
    python scheduler.py --once          # Run once and exit
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from main import run_daily_pipeline, run_scrapers, run_scoring, run_digest
from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Global scheduler
scheduler = None


def job_form4_scraper():
    """Run Form 4 scraper job."""
    logger.info(f"[{datetime.now()}] Running Form 4 scraper...")
    try:
        run_scrapers(form4=True)
    except Exception as e:
        logger.error(f"Form 4 scraper failed: {e}")


def job_13f_scraper():
    """Run 13F scraper job."""
    logger.info(f"[{datetime.now()}] Running 13F scraper...")
    try:
        run_scrapers(form13f=True)
    except Exception as e:
        logger.error(f"13F scraper failed: {e}")


def job_jobs_scraper():
    """Run job posting scraper."""
    logger.info(f"[{datetime.now()}] Running job scraper...")
    try:
        run_scrapers(jobs=True)
    except Exception as e:
        logger.error(f"Job scraper failed: {e}")


def job_signal_scoring():
    """Run signal scoring job."""
    logger.info(f"[{datetime.now()}] Running signal scoring...")
    try:
        run_scoring()
    except Exception as e:
        logger.error(f"Signal scoring failed: {e}")


def job_daily_digest():
    """Generate and send daily digest."""
    logger.info(f"[{datetime.now()}] Running daily digest...")
    try:
        config = get_config()
        send_email = config.email.get('send_daily_digest', False)
        run_digest(send=send_email, save=True)
    except Exception as e:
        logger.error(f"Daily digest failed: {e}")


def job_full_pipeline():
    """Run the complete daily pipeline."""
    logger.info(f"[{datetime.now()}] Running full daily pipeline...")
    try:
        run_daily_pipeline()
    except Exception as e:
        logger.error(f"Daily pipeline failed: {e}")


def setup_scheduler():
    """Setup the scheduler with jobs."""
    global scheduler
    scheduler = BackgroundScheduler()

    config = get_config()

    # Form 4 scraper - every 4 hours during market hours
    scheduler.add_job(
        job_form4_scraper,
        CronTrigger(hour='9,13,17,21', minute=0),  # 9am, 1pm, 5pm, 9pm
        id='form4_scraper',
        name='SEC Form 4 Scraper'
    )

    # 13F scraper - twice daily (quarterly data doesn't change often)
    scheduler.add_job(
        job_13f_scraper,
        CronTrigger(hour='6,18', minute=0),  # 6am and 6pm
        id='form13f_scraper',
        name='SEC 13F Scraper'
    )

    # Job scraper - daily
    scheduler.add_job(
        job_jobs_scraper,
        CronTrigger(hour=8, minute=0),  # 8am daily
        id='jobs_scraper',
        name='Job Posting Scraper'
    )

    # Signal scoring - after scrapers complete
    scheduler.add_job(
        job_signal_scoring,
        CronTrigger(hour='10,14,18,22', minute=30),  # 30 min after Form 4 scraper
        id='signal_scoring',
        name='Signal Scoring'
    )

    # Daily digest - in the morning
    digest_time = config.email.get('digest_time', '07:00')
    hour, minute = map(int, digest_time.split(':'))
    scheduler.add_job(
        job_daily_digest,
        CronTrigger(hour=hour, minute=minute),
        id='daily_digest',
        name='Daily Email Digest'
    )

    return scheduler


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    global scheduler
    logger.info("Received shutdown signal, stopping scheduler...")
    if scheduler:
        scheduler.shutdown()
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(
        description='Insider Signals Scheduler'
    )
    parser.add_argument('--daemon', action='store_true',
                        help='Run as daemon (background)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')

    args = parser.parse_args()

    if args.once:
        logger.info("Running single pipeline execution...")
        job_full_pipeline()
        return

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup and start scheduler
    scheduler = setup_scheduler()
    scheduler.start()

    logger.info("Scheduler started. Jobs scheduled:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")

    if args.daemon:
        logger.info("Running in daemon mode...")
        # In daemon mode, just keep the process alive
        while True:
            time.sleep(60)
    else:
        logger.info("Running in foreground. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            scheduler.shutdown()


if __name__ == '__main__':
    main()
