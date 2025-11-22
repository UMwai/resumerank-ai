"""
SEC EDGAR API scraper for Clinical Trial Signal Detection System.

Fetches 8-K and other filings from biotech companies for signal detection.
Uses the SEC EDGAR API: https://www.sec.gov/edgar/sec-api-documentation
"""
import logging
import re
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database.models import SECFiling, Company

logger = logging.getLogger(__name__)


@dataclass
class EdgarFiling:
    """Parsed SEC EDGAR filing."""
    accession_number: str
    filing_type: str
    filing_date: date
    company_name: str
    cik: str
    description: str
    primary_document: str
    filing_url: str
    items: List[str]  # For 8-K: Item numbers
    raw_content: Optional[str] = None


class SECEdgarScraper:
    """Scraper for SEC EDGAR filings."""

    BASE_URL = "https://data.sec.gov"
    EDGAR_FULL_TEXT = "https://efts.sec.gov/LATEST/search-index"

    # 8-K Item numbers relevant to clinical trials
    RELEVANT_8K_ITEMS = {
        "2.02": "Results of Operations",  # Could mention trial progress
        "7.01": "Regulation FD Disclosure",  # Trial updates often here
        "8.01": "Other Events",  # General announcements including trial news
        "1.01": "Entry into Material Agreement",  # Partnership/licensing deals
        "1.02": "Termination of Material Agreement",  # Negative signal
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.scraper.sec_user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
        self.rate_limit = config.scraper.sec_rate_limit
        self.timeout = config.scraper.request_timeout
        self._last_request = 0

    def _rate_limit_wait(self):
        """Enforce SEC's 10 requests/second rate limit."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _make_request(self, url: str, params: Dict = None) -> requests.Response:
        """Make a rate-limited request to SEC."""
        self._rate_limit_wait()

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"SEC request failed: {e}")
            raise

    def get_company_cik(self, ticker: str) -> Optional[str]:
        """
        Get CIK number for a company ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            CIK number as string (zero-padded to 10 digits) or None
        """
        try:
            # Use SEC's company tickers JSON
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self._make_request(url)
            data = response.json()

            ticker_upper = ticker.upper()
            for entry in data.values():
                if entry.get("ticker") == ticker_upper:
                    cik = str(entry.get("cik_str", ""))
                    return cik.zfill(10)  # Zero-pad to 10 digits

            logger.warning(f"CIK not found for ticker: {ticker}")
            return None

        except Exception as e:
            logger.error(f"Failed to get CIK for {ticker}: {e}")
            return None

    def get_company_filings(
        self,
        cik: str,
        filing_types: List[str] = None,
        days_back: int = 30
    ) -> List[EdgarFiling]:
        """
        Get recent filings for a company.

        Args:
            cik: Company CIK number
            filing_types: List of filing types (e.g., ["8-K", "10-Q"])
            days_back: Number of days to look back

        Returns:
            List of EdgarFiling objects
        """
        if filing_types is None:
            filing_types = ["8-K"]

        # Zero-pad CIK
        cik_padded = cik.zfill(10)

        try:
            # Get submissions for company
            url = f"{self.BASE_URL}/submissions/CIK{cik_padded}.json"
            response = self._make_request(url)
            data = response.json()

            company_name = data.get("name", "Unknown")
            filings_data = data.get("filings", {}).get("recent", {})

            filings = []
            cutoff_date = date.today() - timedelta(days=days_back)

            # Parse recent filings
            forms = filings_data.get("form", [])
            dates = filings_data.get("filingDate", [])
            accessions = filings_data.get("accessionNumber", [])
            primary_docs = filings_data.get("primaryDocument", [])
            descriptions = filings_data.get("primaryDocDescription", [])

            for i in range(len(forms)):
                form_type = forms[i]
                if form_type not in filing_types:
                    continue

                filing_date_str = dates[i]
                filing_date = datetime.strptime(filing_date_str, "%Y-%m-%d").date()

                if filing_date < cutoff_date:
                    continue

                accession = accessions[i]
                accession_clean = accession.replace("-", "")

                filing = EdgarFiling(
                    accession_number=accession,
                    filing_type=form_type,
                    filing_date=filing_date,
                    company_name=company_name,
                    cik=cik_padded,
                    description=descriptions[i] if i < len(descriptions) else "",
                    primary_document=primary_docs[i] if i < len(primary_docs) else "",
                    filing_url=f"https://www.sec.gov/Archives/edgar/data/{cik_padded}/{accession_clean}/{primary_docs[i]}",
                    items=[]
                )

                filings.append(filing)

            logger.info(f"Found {len(filings)} {'/'.join(filing_types)} filings for CIK {cik}")
            return filings

        except Exception as e:
            logger.error(f"Failed to get filings for CIK {cik}: {e}")
            return []

    def get_8k_content(self, filing: EdgarFiling) -> Tuple[str, List[str]]:
        """
        Fetch and parse 8-K filing content.

        Args:
            filing: EdgarFiling object

        Returns:
            Tuple of (raw_text_content, list_of_item_numbers)
        """
        try:
            response = self._make_request(filing.filing_url)
            content = response.text

            # Parse HTML
            soup = BeautifulSoup(content, "html.parser")

            # Extract text
            text = soup.get_text(separator=" ", strip=True)

            # Find 8-K item numbers
            items = self._extract_8k_items(text)

            return text, items

        except Exception as e:
            logger.error(f"Failed to get 8-K content: {e}")
            return "", []

    def _extract_8k_items(self, text: str) -> List[str]:
        """Extract 8-K item numbers from filing text."""
        items = []

        # Common patterns for 8-K items
        patterns = [
            r"Item\s+(\d+\.\d+)",
            r"ITEM\s+(\d+\.\d+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            items.extend(matches)

        # Deduplicate and filter to known items
        unique_items = list(set(items))
        return [item for item in unique_items if item in self.RELEVANT_8K_ITEMS]

    def analyze_8k_for_trial_signals(
        self,
        text: str,
        items: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze 8-K text for clinical trial related signals.

        Args:
            text: Filing text content
            items: List of 8-K item numbers

        Returns:
            List of detected signals
        """
        signals = []
        text_lower = text.lower()

        # Keywords indicating trial-related content
        trial_keywords = [
            "clinical trial", "phase 3", "phase iii", "phase 2", "phase ii",
            "primary endpoint", "secondary endpoint", "efficacy", "safety",
            "enrollment", "fda", "nda", "bla", "pdufa", "breakthrough therapy",
            "accelerated approval", "fast track", "orphan drug",
            "topline results", "data readout", "interim analysis",
        ]

        # Check if filing is trial-related
        is_trial_related = any(kw in text_lower for kw in trial_keywords)

        if not is_trial_related:
            return signals

        # Positive signal keywords
        positive_keywords = {
            "positive": "Positive results mentioned",
            "met primary endpoint": "Primary endpoint met",
            "statistically significant": "Statistical significance achieved",
            "breakthrough therapy": "Breakthrough therapy designation",
            "accelerated approval": "Accelerated approval pathway",
            "fast track": "Fast track designation",
            "exceeded expectations": "Results exceeded expectations",
            "strong efficacy": "Strong efficacy reported",
            "favorable safety": "Favorable safety profile",
        }

        # Negative signal keywords
        negative_keywords = {
            "failed": "Failed trial mentioned",
            "did not meet": "Endpoint not met",
            "terminated": "Trial terminated",
            "discontinued": "Development discontinued",
            "safety concern": "Safety concerns",
            "adverse event": "Adverse events mentioned",
            "not statistically significant": "No statistical significance",
            "clinical hold": "Clinical hold",
        }

        # Check for positive signals
        for keyword, description in positive_keywords.items():
            if keyword in text_lower:
                signals.append({
                    "signal_type": "sec_8k_positive",
                    "description": description,
                    "keyword": keyword,
                    "items": items,
                })

        # Check for negative signals
        for keyword, description in negative_keywords.items():
            if keyword in text_lower:
                signals.append({
                    "signal_type": "sec_8k_negative",
                    "description": description,
                    "keyword": keyword,
                    "items": items,
                })

        return signals

    def fetch_filings_for_companies(
        self,
        tickers: List[str] = None,
        days_back: int = 7
    ) -> List[Tuple[str, EdgarFiling, List[Dict]]]:
        """
        Fetch 8-K filings for multiple companies and analyze them.

        Args:
            tickers: List of stock tickers (uses database if None)
            days_back: Number of days to look back

        Returns:
            List of tuples: (ticker, filing, signals)
        """
        if tickers is None:
            companies = Company.get_all()
            tickers = [c.ticker for c in companies]

        results = []

        for ticker in tickers:
            logger.info(f"Fetching filings for {ticker}...")

            # Get CIK
            company = Company.get_by_ticker(ticker)
            cik = company.cik if company else None

            if not cik:
                cik = self.get_company_cik(ticker)
                if cik and company:
                    company.cik = cik
                    company.save()

            if not cik:
                logger.warning(f"Could not find CIK for {ticker}")
                continue

            # Get filings
            filings = self.get_company_filings(cik, ["8-K"], days_back)

            for filing in filings:
                # Check if already processed
                if SECFiling.exists(filing.accession_number):
                    logger.debug(f"Filing {filing.accession_number} already processed")
                    continue

                # Get content and analyze
                text, items = self.get_8k_content(filing)
                signals = self.analyze_8k_for_trial_signals(text, items)

                # Store filing
                sec_filing = SECFiling(
                    company_ticker=ticker,
                    filing_type=filing.filing_type,
                    filing_date=filing.filing_date,
                    accession_number=filing.accession_number,
                    filing_url=filing.filing_url,
                    description=filing.description,
                    raw_content=text[:10000] if text else None,  # Truncate
                    processed=True
                )
                sec_filing.save()

                if signals:
                    results.append((ticker, filing, signals))
                    logger.info(f"Found {len(signals)} signals in {ticker} 8-K")

        return results


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)

    scraper = SECEdgarScraper()

    # Test with a known biotech company
    print("Testing SEC EDGAR scraper...")

    # Get CIK for Moderna
    cik = scraper.get_company_cik("MRNA")
    print(f"Moderna CIK: {cik}")

    if cik:
        print("\nFetching recent 8-K filings...")
        filings = scraper.get_company_filings(cik, ["8-K"], days_back=30)

        for filing in filings[:3]:
            print(f"\n{filing.filing_date}: {filing.filing_type}")
            print(f"  Accession: {filing.accession_number}")
            print(f"  URL: {filing.filing_url}")

            # Get content
            text, items = scraper.get_8k_content(filing)
            if items:
                print(f"  Items: {', '.join(items)}")

            # Analyze for signals
            signals = scraper.analyze_8k_for_trial_signals(text, items)
            if signals:
                print(f"  Signals found: {len(signals)}")
                for sig in signals:
                    print(f"    - {sig['signal_type']}: {sig['description']}")
