#!/usr/bin/env python3
"""
Scheduler for Insider Activity + Hiring Signals System

Runs scrapers and scoring on a schedule using APScheduler.
Supports both traditional batch scraping and real-time RSS monitoring.

Usage:
    python scheduler.py                 # Run in foreground
    python scheduler.py --daemon        # Run as daemon (background)
    python scheduler.py --once          # Run once and exit
    python scheduler.py --realtime      # Enable real-time RSS monitoring (30-min intervals)
"""

import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from main import run_daily_pipeline, run_scrapers, run_scoring, run_digest
from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Global scheduler
scheduler = None
realtime_monitor = None


def job_form4_scraper():
    """Run Form 4 scraper job (batch mode)."""
    logger.info(f"[{datetime.now()}] Running Form 4 scraper (batch)...")
    try:
        run_scrapers(form4=True)
    except Exception as e:
        logger.error(f"Form 4 scraper failed: {e}")


def job_form4_realtime():
    """Run Form 4 real-time RSS poll."""
    logger.info(f"[{datetime.now()}] Polling Form 4 RSS feed...")
    try:
        from scrapers.form4_realtime import poll_form4_rss
        result = poll_form4_rss()
        logger.info(f"RSS poll: {result.get('filings_processed', 0)} watchlist filings processed")
    except Exception as e:
        logger.error(f"Form 4 RSS poll failed: {e}")


def job_form8k_scraper():
    """Run Form 8-K executive changes scraper."""
    logger.info(f"[{datetime.now()}] Running Form 8-K scraper...")
    try:
        from scrapers.form8k_scraper import Form8KScraper
        scraper = Form8KScraper(use_ai=True)
        result = scraper.run(days=7)
        logger.info(f"8-K scraper: {result.get('changes_found', 0)} executive changes found")
    except Exception as e:
        logger.error(f"Form 8-K scraper failed: {e}")


def job_glassdoor_scraper():
    """Run Glassdoor sentiment scraper."""
    logger.info(f"[{datetime.now()}] Running Glassdoor scraper...")
    try:
        from scrapers.glassdoor_scraper import GlassdoorScraper
        scraper = GlassdoorScraper()
        result = scraper.run(max_pages=2, max_reviews=15)
        logger.info(f"Glassdoor scraper: {result.get('reviews_found', 0)} reviews analyzed")
    except Exception as e:
        logger.error(f"Glassdoor scraper failed: {e}")


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


def setup_scheduler(enable_realtime: bool = False, realtime_interval: int = 30):
    """
    Setup the scheduler with jobs.

    Args:
        enable_realtime: Enable real-time Form 4 RSS monitoring (30-min intervals)
        realtime_interval: Minutes between real-time RSS polls (default: 30)
    """
    global scheduler
    scheduler = BackgroundScheduler()

    config = get_config()

    if enable_realtime:
        # Real-time Form 4 RSS monitoring (every 30 minutes)
        scheduler.add_job(
            job_form4_realtime,
            IntervalTrigger(minutes=realtime_interval),
            id='form4_realtime',
            name='Form 4 Real-Time RSS Monitor'
        )
        logger.info(f"Real-time Form 4 monitoring enabled (every {realtime_interval} minutes)")
    else:
        # Batch Form 4 scraper - every 4 hours during market hours
        scheduler.add_job(
            job_form4_scraper,
            CronTrigger(hour='9,13,17,21', minute=0),  # 9am, 1pm, 5pm, 9pm
            id='form4_scraper',
            name='SEC Form 4 Scraper (Batch)'
        )

    # Form 8-K executive changes scraper - twice daily
    scheduler.add_job(
        job_form8k_scraper,
        CronTrigger(hour='7,19', minute=0),  # 7am and 7pm
        id='form8k_scraper',
        name='SEC Form 8-K Executive Changes Scraper'
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

    # Glassdoor sentiment scraper - weekly (Sunday 6am)
    # Less frequent due to rate limiting concerns
    scheduler.add_job(
        job_glassdoor_scraper,
        CronTrigger(day_of_week='sun', hour=6, minute=0),
        id='glassdoor_scraper',
        name='Glassdoor Sentiment Scraper'
    )

    # Signal scoring - after scrapers complete
    if enable_realtime:
        # More frequent scoring with real-time data
        scheduler.add_job(
            job_signal_scoring,
            CronTrigger(hour='8,10,12,14,16,18,20', minute=0),
            id='signal_scoring',
            name='Signal Scoring'
        )
    else:
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
    parser.add_argument('--realtime', action='store_true',
                        help='Enable real-time Form 4 RSS monitoring (30-min intervals)')
    parser.add_argument('--interval', type=int, default=30,
                        help='Real-time polling interval in minutes (default: 30)')

    args = parser.parse_args()

    if args.once:
        logger.info("Running single pipeline execution...")
        job_full_pipeline()
        return

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup and start scheduler
    scheduler = setup_scheduler(
        enable_realtime=args.realtime,
        realtime_interval=args.interval
    )
    scheduler.start()

    logger.info("Scheduler started. Jobs scheduled:")
    for job in scheduler.get_jobs():
        logger.info(f"  - {job.name}: {job.trigger}")

    if args.realtime:
        logger.info(f"Real-time monitoring enabled (polling every {args.interval} minutes)")

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
