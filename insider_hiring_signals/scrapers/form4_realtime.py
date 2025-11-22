"""
Real-Time SEC Form 4 RSS Feed Monitor

Monitors SEC EDGAR RSS feeds for new Form 4 filings at configurable intervals
(default: 30 minutes). Provides near real-time insider trading alerts.

The SEC provides RSS feeds for latest filings at:
https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=4&company=&dateb=&owner=include&count=100&output=atom

This module:
1. Polls the SEC RSS feed at configurable intervals
2. Filters for watchlist companies
3. Parses new filings immediately
4. Triggers alerts for significant transactions
"""

import hashlib
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set
from threading import Thread, Event
import json

import requests
from bs4 import BeautifulSoup

from utils.config import get_config
from utils.database import get_database
from utils.logger import setup_logger

# Import the base Form4Scraper for parsing functionality
from scrapers.form4_scraper import Form4Scraper, InsiderTransaction

logger = setup_logger(__name__)


@dataclass
class RSSFiling:
    """Represents a Form 4 filing from the RSS feed."""
    accession_number: str
    filing_url: str
    company_name: str
    company_cik: str
    filer_name: str
    filer_cik: str
    filing_date: datetime
    form_type: str
    title: str
    updated: datetime

    def __hash__(self):
        return hash(self.accession_number)

    def __eq__(self, other):
        if isinstance(other, RSSFiling):
            return self.accession_number == other.accession_number
        return False


@dataclass
class AlertConfig:
    """Configuration for real-time alerts."""
    min_transaction_value: float = 50000
    alert_on_ceo_trades: bool = True
    alert_on_multiple_insiders: bool = True
    alert_on_large_trades: bool = True
    large_trade_threshold: float = 500000
    callback: Optional[Callable] = None  # Alert callback function


class Form4RealtimeMonitor:
    """
    Real-time SEC Form 4 RSS feed monitor.

    Features:
    - Polls SEC RSS feed at configurable intervals (default 30 minutes)
    - Maintains seen filings to avoid duplicate processing
    - Filters for watchlist companies only
    - Parses new filings immediately upon detection
    - Triggers configurable alerts for significant transactions
    - Thread-safe for background operation
    """

    # SEC RSS feed URLs
    LATEST_FILINGS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    COMPANY_RSS_URL = "https://www.sec.gov/cgi-bin/browse-edgar"

    # Rate limiting (SEC allows 10 requests/second)
    MIN_REQUEST_INTERVAL = 0.1

    def __init__(
        self,
        config_path: Optional[str] = None,
        poll_interval_minutes: int = 30,
        alert_config: Optional[AlertConfig] = None
    ):
        """
        Initialize the real-time monitor.

        Args:
            config_path: Path to configuration file
            poll_interval_minutes: Minutes between RSS feed polls (default: 30)
            alert_config: Configuration for alerts
        """
        self.config = get_config(config_path)
        self.db = get_database(config_path)
        self.poll_interval = poll_interval_minutes * 60  # Convert to seconds
        self.alert_config = alert_config or AlertConfig()

        # Initialize the base scraper for parsing
        self.form4_scraper = Form4Scraper(config_path)

        # Session for HTTP requests
        self.session = self._create_session()
        self.last_request_time = 0

        # Track seen filings to avoid duplicates
        self._seen_filings: Set[str] = set()
        self._load_recent_filings()

        # Thread control
        self._stop_event = Event()
        self._monitor_thread: Optional[Thread] = None

        # Build CIK to ticker mapping for quick lookups
        self._cik_to_ticker: Dict[str, str] = {}
        self._build_cik_mapping()

        # Statistics
        self.stats = {
            'polls_completed': 0,
            'filings_found': 0,
            'filings_processed': 0,
            'alerts_triggered': 0,
            'errors': 0,
            'last_poll_time': None,
            'last_filing_time': None
        }

    def _create_session(self) -> requests.Session:
        """Create a requests session with proper SEC headers."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': self.config.sec_user_agent,
            'Accept': 'application/atom+xml, application/xml, text/xml',
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        })
        return session

    def _rate_limit_wait(self):
        """Ensure we don't exceed SEC's rate limit."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.MIN_REQUEST_INTERVAL:
            time.sleep(self.MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.time()

    def _make_request(self, url: str, params: Dict = None) -> Optional[requests.Response]:
        """Make a rate-limited request to SEC."""
        self._rate_limit_wait()
        try:
            response = self.session.get(
                url,
                params=params,
                timeout=self.config.scraping.get('timeout_seconds', 30)
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            self.stats['errors'] += 1
            return None

    def _build_cik_mapping(self):
        """Build CIK to ticker mapping for watchlist companies."""
        self._cik_to_ticker = {}

        # Use the form4_scraper's mapping
        self.form4_scraper.load_ticker_cik_mapping()
        ticker_to_cik = self.form4_scraper._ticker_to_cik

        # Reverse the mapping for our watchlist
        for ticker in self.config.watchlist:
            cik = ticker_to_cik.get(ticker)
            if cik:
                # Store with and without leading zeros
                self._cik_to_ticker[cik] = ticker
                self._cik_to_ticker[cik.lstrip('0')] = ticker

        logger.info(f"Built CIK mapping for {len(self._cik_to_ticker) // 2} watchlist companies")

    def _load_recent_filings(self):
        """Load recently processed filings from database to avoid duplicates."""
        try:
            recent = self.db.execute("""
                SELECT DISTINCT filing_url
                FROM insider_transactions
                WHERE filing_date >= %s
            """, (datetime.now().date() - timedelta(days=7),))

            for row in recent:
                # Extract accession number from URL
                url = row.get('filing_url', '')
                if url:
                    # Create a hash of the URL for tracking
                    filing_hash = hashlib.md5(url.encode()).hexdigest()[:16]
                    self._seen_filings.add(filing_hash)

            logger.info(f"Loaded {len(self._seen_filings)} recent filings to skip")
        except Exception as e:
            logger.warning(f"Could not load recent filings: {e}")

    def fetch_latest_rss(self, count: int = 100) -> List[RSSFiling]:
        """
        Fetch the latest Form 4 filings from SEC RSS feed.

        Args:
            count: Number of filings to fetch (max 100)

        Returns:
            List of RSSFiling objects
        """
        params = {
            'action': 'getcurrent',
            'type': '4',
            'company': '',
            'dateb': '',
            'owner': 'include',
            'count': min(count, 100),
            'output': 'atom'
        }

        response = self._make_request(self.LATEST_FILINGS_URL, params)
        if not response:
            return []

        return self._parse_rss_feed(response.text)

    def fetch_company_rss(self, cik: str, count: int = 40) -> List[RSSFiling]:
        """
        Fetch recent Form 4 filings for a specific company.

        Args:
            cik: Company CIK number
            count: Number of filings to fetch

        Returns:
            List of RSSFiling objects
        """
        params = {
            'action': 'getcompany',
            'CIK': cik.zfill(10),
            'type': '4',
            'dateb': '',
            'owner': 'include',
            'count': count,
            'output': 'atom'
        }

        response = self._make_request(self.COMPANY_RSS_URL, params)
        if not response:
            return []

        return self._parse_rss_feed(response.text)

    def _parse_rss_feed(self, xml_content: str) -> List[RSSFiling]:
        """Parse SEC EDGAR Atom feed into RSSFiling objects."""
        filings = []

        try:
            root = ET.fromstring(xml_content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}

            for entry in root.findall('atom:entry', ns):
                filing = self._parse_entry(entry, ns)
                if filing:
                    filings.append(filing)

        except ET.ParseError as e:
            logger.error(f"Failed to parse RSS feed: {e}")
            self.stats['errors'] += 1

        return filings

    def _parse_entry(self, entry: ET.Element, ns: Dict) -> Optional[RSSFiling]:
        """Parse a single RSS feed entry."""
        try:
            title_elem = entry.find('atom:title', ns)
            link_elem = entry.find('atom:link', ns)
            updated_elem = entry.find('atom:updated', ns)
            summary_elem = entry.find('atom:summary', ns)

            if title_elem is None or link_elem is None:
                return None

            title = title_elem.text or ''
            href = link_elem.get('href', '')

            # Extract accession number from URL
            accession_match = re.search(r'data/(\d+)/(\d+-\d+-\d+)', href)
            if not accession_match:
                return None

            filer_cik = accession_match.group(1)
            accession_number = accession_match.group(2)

            # Parse updated time
            updated = datetime.now()
            if updated_elem is not None and updated_elem.text:
                try:
                    updated = datetime.fromisoformat(
                        updated_elem.text.replace('Z', '+00:00')
                    ).replace(tzinfo=None)
                except ValueError:
                    pass

            # Extract company info from title
            # Format: "4 - Company Name (0001234567) (Filer Name)"
            company_name = ''
            company_cik = ''
            filer_name = ''

            # Parse title for company info
            title_match = re.match(r'4\s*-\s*(.+?)\s*\((\d+)\)\s*\((.+?)\)', title)
            if title_match:
                company_name = title_match.group(1).strip()
                company_cik = title_match.group(2)
                filer_name = title_match.group(3).strip()
            else:
                # Fallback: just extract what we can
                company_name = title.replace('4 -', '').strip()

            return RSSFiling(
                accession_number=accession_number,
                filing_url=href,
                company_name=company_name,
                company_cik=company_cik,
                filer_name=filer_name,
                filer_cik=filer_cik,
                filing_date=updated,
                form_type='4',
                title=title,
                updated=updated
            )

        except Exception as e:
            logger.error(f"Failed to parse RSS entry: {e}")
            return None

    def _is_watchlist_company(self, filing: RSSFiling) -> bool:
        """Check if filing is for a watchlist company."""
        # Check by CIK
        if filing.company_cik in self._cik_to_ticker:
            return True
        if filing.company_cik.lstrip('0') in self._cik_to_ticker:
            return True

        # Fallback: check by company name matching
        company_name_lower = filing.company_name.lower()
        for ticker in self.config.watchlist:
            if ticker.lower() in company_name_lower:
                return True

        return False

    def _get_ticker_for_filing(self, filing: RSSFiling) -> Optional[str]:
        """Get ticker symbol for a filing."""
        cik = filing.company_cik.lstrip('0')
        return self._cik_to_ticker.get(cik) or self._cik_to_ticker.get(filing.company_cik)

    def _is_new_filing(self, filing: RSSFiling) -> bool:
        """Check if we've already processed this filing."""
        filing_hash = hashlib.md5(filing.filing_url.encode()).hexdigest()[:16]
        return filing_hash not in self._seen_filings

    def _mark_filing_seen(self, filing: RSSFiling):
        """Mark a filing as processed."""
        filing_hash = hashlib.md5(filing.filing_url.encode()).hexdigest()[:16]
        self._seen_filings.add(filing_hash)

        # Limit memory usage
        if len(self._seen_filings) > 10000:
            # Keep only most recent
            self._seen_filings = set(list(self._seen_filings)[-5000:])

    def process_filing(self, filing: RSSFiling) -> List[InsiderTransaction]:
        """
        Process a single RSS filing.

        Args:
            filing: The RSSFiling to process

        Returns:
            List of parsed InsiderTransaction objects
        """
        ticker = self._get_ticker_for_filing(filing)
        if not ticker:
            logger.warning(f"Could not determine ticker for {filing.company_name}")
            return []

        logger.info(f"Processing Form 4: {ticker} - {filing.filer_name}")

        try:
            # Get the XML URL from the filing page
            xml_url = self.form4_scraper.get_form4_xml_url(filing.filing_url)
            if not xml_url:
                logger.warning(f"Could not find XML for {filing.filing_url}")
                return []

            # Parse the Form 4 XML
            transactions = self.form4_scraper.parse_form4_xml(
                xml_url, ticker, filing.filing_date.date()
            )

            if transactions:
                # Save to database
                inserted, _ = self.form4_scraper.save_transactions(transactions)
                logger.info(f"Saved {inserted} transactions for {ticker}")

                # Check for alerts
                self._check_alerts(transactions, filing)

            self.stats['filings_processed'] += 1
            self.stats['last_filing_time'] = datetime.now()

            return transactions

        except Exception as e:
            logger.error(f"Error processing filing {filing.accession_number}: {e}")
            self.stats['errors'] += 1
            return []

    def _check_alerts(self, transactions: List[InsiderTransaction], filing: RSSFiling):
        """Check transactions against alert criteria."""
        for trans in transactions:
            should_alert = False
            alert_reasons = []

            # CEO trades
            if self.alert_config.alert_on_ceo_trades:
                title = (trans.insider_title or '').upper()
                if 'CEO' in title or 'CHIEF EXECUTIVE' in title:
                    should_alert = True
                    alert_reasons.append(f"CEO {'purchased' if trans.transaction_type == 'Purchase' else 'sold'} shares")

            # Large trades
            if self.alert_config.alert_on_large_trades:
                if trans.transaction_value >= self.alert_config.large_trade_threshold:
                    should_alert = True
                    alert_reasons.append(f"Large trade: ${trans.transaction_value:,.0f}")

            # Minimum value check
            if trans.transaction_value < self.alert_config.min_transaction_value:
                should_alert = False

            if should_alert:
                self._trigger_alert(trans, alert_reasons, filing)

    def _trigger_alert(
        self,
        transaction: InsiderTransaction,
        reasons: List[str],
        filing: RSSFiling
    ):
        """Trigger an alert for a significant transaction."""
        self.stats['alerts_triggered'] += 1

        alert_data = {
            'ticker': transaction.company_ticker,
            'insider': transaction.insider_name,
            'title': transaction.insider_title,
            'transaction_type': transaction.transaction_type,
            'shares': transaction.shares,
            'price': transaction.price_per_share,
            'value': transaction.transaction_value,
            'date': str(transaction.transaction_date),
            'filing_url': filing.filing_url,
            'reasons': reasons,
            'is_10b5_1': transaction.is_10b5_1_plan,
            'timestamp': datetime.now().isoformat()
        }

        logger.warning(
            f"ALERT: {transaction.company_ticker} - {transaction.insider_name} "
            f"({transaction.insider_title}) {transaction.transaction_type} "
            f"${transaction.transaction_value:,.0f} - {', '.join(reasons)}"
        )

        # Call custom callback if configured
        if self.alert_config.callback:
            try:
                self.alert_config.callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def poll_once(self) -> Dict[str, Any]:
        """
        Perform a single poll of the RSS feed.

        Returns:
            Dictionary with poll results
        """
        logger.info("Polling SEC Form 4 RSS feed...")

        results = {
            'filings_found': 0,
            'filings_processed': 0,
            'transactions_saved': 0,
            'alerts': 0,
            'watchlist_filings': [],
            'timestamp': datetime.now().isoformat()
        }

        try:
            # Fetch latest filings
            filings = self.fetch_latest_rss(count=100)
            results['filings_found'] = len(filings)
            self.stats['filings_found'] += len(filings)

            # Filter for watchlist and new filings
            for filing in filings:
                if not self._is_watchlist_company(filing):
                    continue

                if not self._is_new_filing(filing):
                    continue

                # Process the filing
                transactions = self.process_filing(filing)

                if transactions:
                    results['filings_processed'] += 1
                    results['transactions_saved'] += len(transactions)
                    results['watchlist_filings'].append({
                        'ticker': self._get_ticker_for_filing(filing),
                        'company': filing.company_name,
                        'filer': filing.filer_name,
                        'transactions': len(transactions)
                    })

                # Mark as seen
                self._mark_filing_seen(filing)

            self.stats['polls_completed'] += 1
            self.stats['last_poll_time'] = datetime.now()

            logger.info(
                f"Poll complete: {results['filings_found']} total, "
                f"{results['filings_processed']} watchlist, "
                f"{results['transactions_saved']} transactions saved"
            )

        except Exception as e:
            logger.error(f"Poll failed: {e}")
            self.stats['errors'] += 1
            results['error'] = str(e)

        return results

    def _monitor_loop(self):
        """Background monitoring loop."""
        logger.info(f"Starting real-time monitor (polling every {self.poll_interval // 60} minutes)")

        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                self.stats['errors'] += 1

            # Wait for next poll or stop signal
            self._stop_event.wait(timeout=self.poll_interval)

        logger.info("Real-time monitor stopped")

    def start(self, background: bool = True):
        """
        Start the real-time monitor.

        Args:
            background: If True, run in background thread. If False, block.
        """
        if self._monitor_thread and self._monitor_thread.is_alive():
            logger.warning("Monitor is already running")
            return

        self._stop_event.clear()

        if background:
            self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("Real-time monitor started in background")
        else:
            self._monitor_loop()

    def stop(self):
        """Stop the real-time monitor."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
            self._monitor_thread = None
        logger.info("Real-time monitor stopped")

    def is_running(self) -> bool:
        """Check if the monitor is currently running."""
        return self._monitor_thread is not None and self._monitor_thread.is_alive()

    def get_stats(self) -> Dict[str, Any]:
        """Get monitor statistics."""
        return {
            **self.stats,
            'is_running': self.is_running(),
            'poll_interval_minutes': self.poll_interval // 60,
            'seen_filings_count': len(self._seen_filings),
            'watchlist_count': len(self.config.watchlist)
        }


# Convenience functions for scheduler integration

def start_realtime_monitor(
    poll_interval_minutes: int = 30,
    alert_callback: Optional[Callable] = None
) -> Form4RealtimeMonitor:
    """
    Start a real-time Form 4 monitor.

    Args:
        poll_interval_minutes: Minutes between polls (default 30)
        alert_callback: Function to call on alerts

    Returns:
        Running Form4RealtimeMonitor instance
    """
    alert_config = AlertConfig(callback=alert_callback)
    monitor = Form4RealtimeMonitor(
        poll_interval_minutes=poll_interval_minutes,
        alert_config=alert_config
    )
    monitor.start(background=True)
    return monitor


def poll_form4_rss() -> Dict[str, Any]:
    """
    Single poll of Form 4 RSS feed.

    Convenience function for scheduler integration.

    Returns:
        Poll results dictionary
    """
    monitor = Form4RealtimeMonitor()
    return monitor.poll_once()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Real-time Form 4 RSS Monitor')
    parser.add_argument('--interval', type=int, default=30,
                        help='Poll interval in minutes (default: 30)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    parser.add_argument('--background', action='store_true',
                        help='Run in background mode')

    args = parser.parse_args()

    if args.once:
        # Single poll
        monitor = Form4RealtimeMonitor()
        results = monitor.poll_once()
        print(f"\nPoll Results:")
        print(f"  Filings found: {results['filings_found']}")
        print(f"  Watchlist filings: {results['filings_processed']}")
        print(f"  Transactions saved: {results['transactions_saved']}")
        if results.get('watchlist_filings'):
            print(f"\n  Watchlist Filings:")
            for f in results['watchlist_filings']:
                print(f"    - {f['ticker']}: {f['filer']} ({f['transactions']} transactions)")
    else:
        # Continuous monitoring
        def alert_handler(alert_data):
            print(f"\n*** ALERT: {alert_data['ticker']} - {alert_data['insider']} "
                  f"{alert_data['transaction_type']} ${alert_data['value']:,.0f} ***\n")

        monitor = start_realtime_monitor(
            poll_interval_minutes=args.interval,
            alert_callback=alert_handler
        )

        print(f"Real-time monitor started (polling every {args.interval} minutes)")
        print("Press Ctrl+C to stop")

        try:
            while monitor.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            monitor.stop()

            stats = monitor.get_stats()
            print(f"\nFinal Stats:")
            print(f"  Polls completed: {stats['polls_completed']}")
            print(f"  Filings found: {stats['filings_found']}")
            print(f"  Filings processed: {stats['filings_processed']}")
            print(f"  Alerts triggered: {stats['alerts_triggered']}")
            print(f"  Errors: {stats['errors']}")
