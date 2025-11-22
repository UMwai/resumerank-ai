#!/usr/bin/env python3
"""
Insider Activity + Hiring Signals System
Main entry point for running scrapers, scoring, and reports.

Usage:
    python main.py scrape [--form4] [--13f] [--jobs] [--all]
    python main.py score [--ticker TICKER]
    python main.py report [--console] [--html] [--json]
    python main.py digest [--send] [--save]
    python main.py run-daily
"""

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scrapers import Form4Scraper, Form13FScraper, JobScraper
from models import SignalScorer
from reports import EmailDigest, ReportGenerator
from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

logger = setup_logger(__name__)


def run_scrapers(form4: bool = False, form13f: bool = False,
                 jobs: bool = False, all_scrapers: bool = False):
    """Run specified scrapers."""
    results = {}

    if all_scrapers:
        form4 = form13f = jobs = True

    if form4:
        logger.info("Running Form 4 scraper...")
        scraper = Form4Scraper()
        results['form4'] = scraper.run()
        logger.info(f"Form 4 complete: {results['form4']}")

    if form13f:
        logger.info("Running 13F scraper...")
        scraper = Form13FScraper()
        results['form13f'] = scraper.run()
        logger.info(f"13F complete: {results['form13f']}")

    if jobs:
        logger.info("Running job scraper...")
        scraper = JobScraper()
        results['jobs'] = scraper.run()
        logger.info(f"Jobs complete: {results['jobs']}")

    return results


def run_scoring(ticker: str = None):
    """Run signal scoring for specified or all companies."""
    scorer = SignalScorer()

    if ticker:
        logger.info(f"Scoring {ticker}...")
        score = scorer.calculate_score(ticker)
        scorer.save_score(score)
        print(f"\n{ticker}: Score={score.composite_score:.2f}, "
              f"Confidence={score.confidence:.2f}, "
              f"Recommendation={score.recommendation}")
        print(f"  Signals: {score.signal_count}")
        print(f"  Insider: {score.insider_score:.2f}")
        print(f"  Institutional: {score.institutional_score:.2f}")
        print(f"  Hiring: {score.hiring_score:.2f}")
        return score
    else:
        logger.info("Scoring all watchlist companies...")
        scores = scorer.score_all_companies()
        logger.info(f"Scored {len(scores)} companies")
        return scores


def run_report(console: bool = False, html: bool = False,
               json_export: bool = False, example: bool = False):
    """Generate reports in specified formats."""
    generator = ReportGenerator()

    if example:
        # Generate example report
        output_file = Path(__file__).parent / 'data' / 'reports' / 'example_signal_report.txt'
        report = generator.generate_example_report(str(output_file))
        print(report)
        return

    if console:
        generator.print_console_report()

    if html:
        # Generate HTML via email digest (reuse template)
        digest = EmailDigest()
        content = digest.generate_digest()
        output_file = Path(__file__).parent / 'data' / 'reports' / f'report_{date.today().isoformat()}.html'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(content['html'])
        logger.info(f"HTML report saved to {output_file}")

    if json_export:
        output_file = Path(__file__).parent / 'data' / 'reports' / f'signals_{date.today().isoformat()}.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        generator.export_json(output_file=str(output_file))
        logger.info(f"JSON export saved to {output_file}")


def run_digest(send: bool = False, save: bool = False):
    """Generate and optionally send daily digest."""
    digest = EmailDigest()
    result = digest.run(send=send, save_html=save)

    if save:
        logger.info("Digest saved to file")
    if send:
        if result['sent']:
            logger.info("Digest email sent successfully")
        else:
            logger.warning("Failed to send digest email")

    return result


def run_daily_pipeline():
    """Run the complete daily pipeline."""
    logger.info("=" * 60)
    logger.info(f"Starting daily pipeline: {datetime.now()}")
    logger.info("=" * 60)

    results = {
        'start_time': datetime.now().isoformat(),
        'scrapers': {},
        'scoring': {},
        'digest': {}
    }

    try:
        # Step 1: Run scrapers
        logger.info("Step 1: Running scrapers...")
        results['scrapers'] = run_scrapers(all_scrapers=True)

        # Step 2: Run scoring
        logger.info("Step 2: Running signal scoring...")
        scores = run_scoring()
        results['scoring'] = {
            'companies_scored': len(scores) if scores else 0,
            'top_score': max(s.composite_score for s in scores) if scores else 0,
            'bottom_score': min(s.composite_score for s in scores) if scores else 0
        }

        # Step 3: Generate and send digest
        logger.info("Step 3: Generating daily digest...")
        config = get_config()
        send_email = config.email.get('send_daily_digest', False)
        results['digest'] = run_digest(send=send_email, save=True)

        results['status'] = 'success'
        results['end_time'] = datetime.now().isoformat()

        logger.info("=" * 60)
        logger.info("Daily pipeline completed successfully")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        results['status'] = 'failed'
        results['error'] = str(e)
        results['end_time'] = datetime.now().isoformat()

    return results


def init_database():
    """Initialize database with schema."""
    schema_path = Path(__file__).parent / 'schema.sql'
    if not schema_path.exists():
        logger.error(f"Schema file not found: {schema_path}")
        return False

    db = get_database()
    with open(schema_path, 'r') as f:
        schema_sql = f.read()

    try:
        # Split by semicolons and execute each statement
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]
        with db.cursor() as cur:
            for statement in statements:
                if statement and not statement.startswith('--'):
                    try:
                        cur.execute(statement)
                    except Exception as e:
                        # Ignore errors for CREATE IF NOT EXISTS etc.
                        if 'already exists' not in str(e):
                            logger.warning(f"Statement warning: {e}")

        logger.info("Database schema initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Insider Activity + Hiring Signals System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all scrapers
    python main.py scrape --all

    # Run only Form 4 scraper
    python main.py scrape --form4

    # Score a specific company
    python main.py score --ticker MRNA

    # Generate console report
    python main.py report --console

    # Generate and send daily digest
    python main.py digest --send

    # Run complete daily pipeline
    python main.py run-daily

    # Initialize database
    python main.py init-db

    # Generate example report
    python main.py report --example
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Scrape command
    scrape_parser = subparsers.add_parser('scrape', help='Run data scrapers')
    scrape_parser.add_argument('--form4', action='store_true', help='Run Form 4 scraper')
    scrape_parser.add_argument('--13f', dest='form13f', action='store_true', help='Run 13F scraper')
    scrape_parser.add_argument('--jobs', action='store_true', help='Run job scraper')
    scrape_parser.add_argument('--all', action='store_true', help='Run all scrapers')

    # Score command
    score_parser = subparsers.add_parser('score', help='Run signal scoring')
    score_parser.add_argument('--ticker', type=str, help='Score specific ticker')

    # Report command
    report_parser = subparsers.add_parser('report', help='Generate reports')
    report_parser.add_argument('--console', action='store_true', help='Print to console')
    report_parser.add_argument('--html', action='store_true', help='Generate HTML report')
    report_parser.add_argument('--json', action='store_true', help='Export as JSON')
    report_parser.add_argument('--example', action='store_true', help='Generate example report')

    # Digest command
    digest_parser = subparsers.add_parser('digest', help='Generate daily digest')
    digest_parser.add_argument('--send', action='store_true', help='Send email')
    digest_parser.add_argument('--save', action='store_true', help='Save to file')

    # Run daily command
    subparsers.add_parser('run-daily', help='Run complete daily pipeline')

    # Init database command
    subparsers.add_parser('init-db', help='Initialize database schema')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'scrape':
        run_scrapers(
            form4=args.form4,
            form13f=args.form13f,
            jobs=args.jobs,
            all_scrapers=args.all
        )

    elif args.command == 'score':
        run_scoring(ticker=args.ticker)

    elif args.command == 'report':
        run_report(
            console=args.console,
            html=args.html,
            json_export=args.json,
            example=args.example
        )

    elif args.command == 'digest':
        run_digest(send=args.send, save=args.save)

    elif args.command == 'run-daily':
        run_daily_pipeline()

    elif args.command == 'init-db':
        init_database()


if __name__ == '__main__':
    main()
