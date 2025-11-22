"""
USPTO Patent Application Scraper for Clinical Trial Signal Detection System.

Monitors patent applications and grants from biotech companies.
Uses USPTO's Open Data Portal API: https://developer.uspto.gov/
"""
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from utils.rate_limiter import get_rate_limiter, rate_limited
from utils.retry import RetryConfig, retry_request, RetryExhausted
from utils.validation import validate_patent, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PatentRecord:
    """Parsed patent or application record from USPTO."""
    patent_number: Optional[str] = None
    application_number: Optional[str] = None
    title: str = ""
    abstract: str = ""
    filing_date: Optional[date] = None
    publication_date: Optional[date] = None
    grant_date: Optional[date] = None
    inventors: List[str] = field(default_factory=list)
    assignee: str = ""
    patent_url: str = ""
    claims_count: int = 0
    patent_type: str = ""  # utility, design, plant
    raw_data: Dict[str, Any] = field(default_factory=dict)


class USPTOScraper:
    """
    Scraper for USPTO patent data.

    Searches for patents and applications related to clinical trials,
    drugs, and biotech companies.
    """

    # USPTO PatentsView API (for granted patents)
    PATENTSVIEW_URL = "https://api.patentsview.org/patents/query"

    # USPTO Open Data Portal
    USPTO_SEARCH_URL = "https://developer.uspto.gov/ibd-api/v1/patent/application"

    # Keywords relevant to clinical trials
    CLINICAL_KEYWORDS = [
        "pharmaceutical", "drug", "therapeutic", "treatment",
        "antibody", "protein", "inhibitor", "receptor",
        "clinical trial", "dosage", "formulation",
    ]

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ClinicalTrialSignals/1.0 (research@example.com)",
            "Accept": "application/json",
        })
        self.timeout = config.scraper.request_timeout
        self.retry_config = RetryConfig(
            max_retries=config.scraper.max_retries,
            base_delay=1.0,
            max_delay=30.0,
        )

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        params: Dict = None,
        json_data: Dict = None,
    ) -> Optional[Dict]:
        """
        Make a rate-limited and retry-enabled request.

        Args:
            url: URL to request
            method: HTTP method
            params: Query parameters
            json_data: JSON body for POST requests

        Returns:
            JSON response or None on failure
        """
        # Apply rate limiting
        limiter = get_rate_limiter("uspto")
        limiter.acquire(timeout=60)

        try:
            response = retry_request(
                method=method,
                url=url,
                session=self.session,
                config=self.retry_config,
                params=params,
                json=json_data,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except RetryExhausted as e:
            logger.error(f"USPTO request failed after retries: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"USPTO request error: {e}")
            return None
        except ValueError as e:
            logger.error(f"Invalid JSON response from USPTO: {e}")
            return None

    def search_patents_by_assignee(
        self,
        assignee_name: str,
        days_back: int = 90,
        limit: int = 25
    ) -> List[PatentRecord]:
        """
        Search for patents by assignee (company) name.

        Uses PatentsView API for granted patents.

        Args:
            assignee_name: Company name to search for
            days_back: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of PatentRecord objects
        """
        cutoff_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Build query for PatentsView API
        query = {
            "_and": [
                {"_contains": {"assignee_organization": assignee_name}},
                {"_gte": {"patent_date": cutoff_date}}
            ]
        }

        params = {
            "q": str(query).replace("'", '"'),
            "f": '["patent_number","patent_title","patent_abstract","patent_date",'
                 '"assignee_organization","inventor_first_name","inventor_last_name",'
                 '"patent_num_claims","app_date"]',
            "o": '{"per_page": ' + str(limit) + '}',
        }

        logger.info(f"Searching USPTO for patents from: {assignee_name}")

        data = self._make_request(self.PATENTSVIEW_URL, params=params)

        if not data:
            return []

        patents = []
        for patent_data in data.get("patents", []):
            patent = self._parse_patentsview_result(patent_data)
            if patent:
                patents.append(patent)

        logger.info(f"Found {len(patents)} patents for {assignee_name}")
        return patents

    def search_patents_by_keywords(
        self,
        keywords: List[str],
        assignee_filter: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[PatentRecord]:
        """
        Search for patents by keywords in title/abstract.

        Args:
            keywords: Keywords to search for
            assignee_filter: Optional assignee name to filter by
            days_back: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of PatentRecord objects
        """
        cutoff_date = (date.today() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        # Build query
        keyword_conditions = [
            {"_or": [
                {"_contains": {"patent_title": kw}},
                {"_contains": {"patent_abstract": kw}}
            ]}
            for kw in keywords
        ]

        query_conditions = [
            {"_gte": {"patent_date": cutoff_date}},
            {"_or": keyword_conditions}
        ]

        if assignee_filter:
            query_conditions.append(
                {"_contains": {"assignee_organization": assignee_filter}}
            )

        query = {"_and": query_conditions}

        params = {
            "q": str(query).replace("'", '"'),
            "f": '["patent_number","patent_title","patent_abstract","patent_date",'
                 '"assignee_organization","inventor_first_name","inventor_last_name",'
                 '"patent_num_claims","app_date"]',
            "o": '{"per_page": ' + str(limit) + '}',
        }

        logger.info(f"Searching USPTO for keywords: {keywords[:3]}...")

        data = self._make_request(self.PATENTSVIEW_URL, params=params)

        if not data:
            return []

        patents = []
        for patent_data in data.get("patents", []):
            patent = self._parse_patentsview_result(patent_data)
            if patent:
                patents.append(patent)

        logger.info(f"Found {len(patents)} patents matching keywords")
        return patents

    def _parse_patentsview_result(self, data: Dict) -> Optional[PatentRecord]:
        """Parse PatentsView API response into PatentRecord."""
        try:
            patent_number = data.get("patent_number", "")

            # Parse inventors
            inventors = []
            for inv in data.get("inventors", []):
                first = inv.get("inventor_first_name", "")
                last = inv.get("inventor_last_name", "")
                if first or last:
                    inventors.append(f"{first} {last}".strip())

            # Parse assignee
            assignees = data.get("assignees", [])
            assignee = assignees[0].get("assignee_organization", "") if assignees else ""

            # Parse dates
            patent_date = self._parse_date(data.get("patent_date"))
            app_date = self._parse_date(data.get("applications", [{}])[0].get("app_date"))

            return PatentRecord(
                patent_number=f"US{patent_number}" if patent_number else None,
                application_number=data.get("applications", [{}])[0].get("app_number"),
                title=data.get("patent_title", ""),
                abstract=data.get("patent_abstract", "")[:2000] if data.get("patent_abstract") else "",
                filing_date=app_date,
                grant_date=patent_date,
                inventors=inventors,
                assignee=assignee,
                claims_count=data.get("patent_num_claims", 0),
                patent_url=f"https://patents.google.com/patent/US{patent_number}" if patent_number else "",
                raw_data=data,
            )

        except Exception as e:
            logger.warning(f"Failed to parse patent result: {e}")
            return None

    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string into date object."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    def search_for_company(
        self,
        company_name: str,
        ticker: Optional[str] = None,
        days_back: int = 90
    ) -> Tuple[List[PatentRecord], List[Dict[str, Any]]]:
        """
        Search for patents related to a company and detect signals.

        Args:
            company_name: Company name to search
            ticker: Optional stock ticker
            days_back: Days to look back

        Returns:
            Tuple of (patents, signals)
        """
        patents = self.search_patents_by_assignee(company_name, days_back)

        signals = []

        # Detect signals from patents
        recent_patents = [
            p for p in patents
            if p.grant_date and (date.today() - p.grant_date).days <= 30
        ]

        for patent in recent_patents:
            # Validate patent data
            validation = validate_patent({
                "patent_number": patent.patent_number,
                "title": patent.title,
                "assignee": patent.assignee,
                "grant_date": patent.grant_date,
            })

            if validation.is_valid:
                signals.append({
                    "signal_type": "patent_filed",
                    "description": f"New patent granted: {patent.title[:100]}",
                    "patent_number": patent.patent_number,
                    "patent_title": patent.title,
                    "grant_date": patent.grant_date.isoformat() if patent.grant_date else None,
                })
                logger.info(f"Patent signal detected for {company_name}: {patent.patent_number}")

        return patents, signals

    def analyze_patent_for_trial_relevance(
        self,
        patent: PatentRecord,
        trial_keywords: List[str] = None
    ) -> float:
        """
        Analyze if a patent is relevant to clinical trials.

        Args:
            patent: PatentRecord to analyze
            trial_keywords: Optional list of keywords to search for

        Returns:
            Relevance score between 0 and 1
        """
        if trial_keywords is None:
            trial_keywords = self.CLINICAL_KEYWORDS

        text = f"{patent.title} {patent.abstract}".lower()

        # Count keyword matches
        matches = sum(1 for kw in trial_keywords if kw.lower() in text)

        # Calculate relevance score
        relevance = min(1.0, matches / 3)  # 3+ keywords = max relevance

        return relevance

    def fetch_patents_for_companies(
        self,
        companies: List[Dict[str, str]],
        days_back: int = 30
    ) -> List[Tuple[str, PatentRecord, List[Dict]]]:
        """
        Fetch patents for multiple companies and detect signals.

        Args:
            companies: List of dicts with 'name' and 'ticker' keys
            days_back: Days to look back

        Returns:
            List of tuples: (ticker, patent, signals)
        """
        results = []

        for company in companies:
            name = company.get("name", "")
            ticker = company.get("ticker", "")

            if not name:
                continue

            logger.info(f"Searching patents for {name} ({ticker})...")

            try:
                patents, signals = self.search_for_company(
                    company_name=name,
                    ticker=ticker,
                    days_back=days_back
                )

                for patent in patents:
                    patent_signals = [s for s in signals if s.get("patent_number") == patent.patent_number]
                    if patent_signals:
                        results.append((ticker, patent, patent_signals))

            except Exception as e:
                logger.error(f"Error fetching patents for {name}: {e}")
                continue

        logger.info(f"Found {len(results)} patent signals across all companies")
        return results


# Company name mappings for patent searches
PATENT_COMPANY_MAPPINGS = {
    "SAVA": ["Cassava Sciences"],
    "MRNA": ["Moderna", "ModernaTX"],
    "NVAX": ["Novavax"],
    "IONS": ["Ionis Pharmaceuticals", "Ionis"],
    "ALNY": ["Alnylam Pharmaceuticals", "Alnylam"],
    "BMRN": ["BioMarin", "BioMarin Pharmaceutical"],
    "SRPT": ["Sarepta Therapeutics", "Sarepta"],
    "RARE": ["Ultragenyx Pharmaceutical", "Ultragenyx"],
    "NBIX": ["Neurocrine Biosciences", "Neurocrine"],
    "ACAD": ["ACADIA Pharmaceuticals", "ACADIA"],
}


if __name__ == "__main__":
    # Test the scraper
    import sys
    logging.basicConfig(level=logging.INFO)

    scraper = USPTOScraper()

    print("Testing USPTO scraper...")
    print("-" * 40)

    # Test with Moderna
    print("\nSearching for Moderna patents...")
    patents = scraper.search_patents_by_assignee("Moderna", days_back=180, limit=5)

    for patent in patents[:3]:
        print(f"\n{patent.patent_number}: {patent.title[:60]}...")
        print(f"  Grant date: {patent.grant_date}")
        print(f"  Assignee: {patent.assignee}")
        print(f"  Claims: {patent.claims_count}")

        # Check relevance
        relevance = scraper.analyze_patent_for_trial_relevance(patent)
        print(f"  Clinical relevance: {relevance:.2f}")

    # Test signal detection
    print("\n\nTesting signal detection...")
    patents, signals = scraper.search_for_company("Moderna", "MRNA", days_back=90)
    print(f"Found {len(patents)} patents, {len(signals)} signals")

    for signal in signals[:3]:
        print(f"  Signal: {signal['signal_type']}")
        print(f"    {signal['description'][:60]}...")
