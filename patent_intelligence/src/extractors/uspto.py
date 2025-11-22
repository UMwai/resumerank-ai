"""
USPTO Patent Data Extractor

Extracts patent information from the USPTO PatentsView API.
Calculates patent term adjustments (PTA), extensions (PTE), and final expiration dates.

Data source: https://patentsview.org/apis/api-endpoints/patents
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil.relativedelta import relativedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PatentInfo:
    """Data class for patent information from USPTO."""

    patent_number: str
    patent_title: str
    patent_type: str
    filing_date: Optional[date]
    grant_date: Optional[date]
    abstract: Optional[str]
    num_claims: int
    assignees: List[str]
    inventors: List[str]
    cpc_codes: List[str]  # Cooperative Patent Classification

    # Calculated fields
    base_expiration_date: Optional[date] = None
    pta_days: int = 0
    pte_days: int = 0
    adjusted_expiration_date: Optional[date] = None


class USPTOExtractor:
    """
    Extracts patent information from USPTO PatentsView API.

    The PatentsView API provides free access to patent data with the following features:
    - Query by patent number, assignee, filing date, etc.
    - Returns patent metadata, claims, citations, etc.
    - Rate limited to 45 requests per minute
    """

    BASE_URL = "https://api.patentsview.org/patents/query"
    RATE_LIMIT = 45  # requests per minute

    # Patent term is 20 years from filing date for patents filed after June 8, 1995
    MODERN_PATENT_TERM_YEARS = 20

    # Pharmaceutical patents may get up to 5 years PTE (Patent Term Extension)
    MAX_PTE_YEARS = 5

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the USPTO extractor.

        Args:
            api_key: Optional API key (not required for basic queries).
        """
        self.api_key = api_key
        self._request_count = 0
        self._last_request_time: Optional[datetime] = None

    def _rate_limit(self) -> None:
        """Implement rate limiting."""
        import time

        if self._last_request_time is not None:
            elapsed = (datetime.now() - self._last_request_time).total_seconds()
            if elapsed < 60 / self.RATE_LIMIT:
                sleep_time = (60 / self.RATE_LIMIT) - elapsed
                time.sleep(sleep_time)

        self._last_request_time = datetime.now()
        self._request_count += 1

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _make_request(self, query: Dict, fields: List[str]) -> Dict:
        """
        Make a request to the PatentsView API.

        Args:
            query: Query parameters.
            fields: Fields to return.

        Returns:
            API response as dictionary.
        """
        self._rate_limit()

        params = {
            "q": query,
            "f": fields,
            "o": {"per_page": 100},
        }

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-Api-Key"] = self.api_key

        response = requests.post(
            self.BASE_URL,
            json=params,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()

        return response.json()

    def get_patent_by_number(self, patent_number: str) -> Optional[PatentInfo]:
        """
        Get patent information by patent number.

        Args:
            patent_number: USPTO patent number.

        Returns:
            PatentInfo object or None if not found.
        """
        # Clean patent number (remove leading zeros, commas, etc.)
        patent_number = patent_number.replace(",", "").strip()

        query = {"patent_number": patent_number}

        fields = [
            "patent_number",
            "patent_title",
            "patent_type",
            "patent_date",
            "patent_abstract",
            "patent_num_claims",
            "patent_firstnamed_assignee_id",
            "assignees",
            "inventors",
        ]

        try:
            response = self._make_request(query, fields)

            if not response.get("patents"):
                logger.warning(f"Patent {patent_number} not found in USPTO database")
                return None

            patent_data = response["patents"][0]
            return self._parse_patent_response(patent_data)

        except requests.RequestException as e:
            logger.error(f"Error fetching patent {patent_number}: {e}")
            return None

    def get_patents_by_numbers(
        self, patent_numbers: List[str]
    ) -> Dict[str, Optional[PatentInfo]]:
        """
        Get multiple patents by their numbers.

        Args:
            patent_numbers: List of USPTO patent numbers.

        Returns:
            Dictionary mapping patent numbers to PatentInfo objects.
        """
        results = {}

        # Process in batches
        batch_size = 25
        for i in range(0, len(patent_numbers), batch_size):
            batch = patent_numbers[i : i + batch_size]

            # Clean patent numbers
            clean_batch = [p.replace(",", "").strip() for p in batch]

            query = {"_or": [{"patent_number": p} for p in clean_batch]}

            fields = [
                "patent_number",
                "patent_title",
                "patent_type",
                "patent_date",
                "patent_abstract",
                "patent_num_claims",
                "assignees",
                "inventors",
            ]

            try:
                response = self._make_request(query, fields)

                for patent_data in response.get("patents", []):
                    patent_info = self._parse_patent_response(patent_data)
                    if patent_info:
                        results[patent_info.patent_number] = patent_info

            except requests.RequestException as e:
                logger.error(f"Error fetching patent batch: {e}")

        # Fill in None for patents not found
        for patent_number in patent_numbers:
            clean_number = patent_number.replace(",", "").strip()
            if clean_number not in results:
                results[clean_number] = None

        logger.info(
            f"Retrieved {len([v for v in results.values() if v])} of "
            f"{len(patent_numbers)} patents from USPTO"
        )

        return results

    def get_patents_by_assignee(
        self, assignee_name: str, limit: int = 100
    ) -> List[PatentInfo]:
        """
        Get patents assigned to a specific company.

        Args:
            assignee_name: Company name to search.
            limit: Maximum number of patents to return.

        Returns:
            List of PatentInfo objects.
        """
        query = {
            "_and": [
                {"_contains": {"assignee_organization": assignee_name}},
                {"patent_type": "utility"},  # Utility patents only
            ]
        }

        fields = [
            "patent_number",
            "patent_title",
            "patent_type",
            "patent_date",
            "patent_abstract",
            "patent_num_claims",
            "assignees",
            "inventors",
        ]

        try:
            response = self._make_request(query, fields)

            patents = []
            for patent_data in response.get("patents", [])[:limit]:
                patent_info = self._parse_patent_response(patent_data)
                if patent_info:
                    patents.append(patent_info)

            logger.info(f"Found {len(patents)} patents for assignee: {assignee_name}")
            return patents

        except requests.RequestException as e:
            logger.error(f"Error fetching patents for {assignee_name}: {e}")
            return []

    def _parse_patent_response(self, data: Dict) -> Optional[PatentInfo]:
        """
        Parse API response into PatentInfo object.

        Args:
            data: Patent data from API response.

        Returns:
            PatentInfo object.
        """
        try:
            # Parse grant date
            grant_date = None
            if data.get("patent_date"):
                grant_date = datetime.strptime(data["patent_date"], "%Y-%m-%d").date()

            # Extract assignees
            assignees = []
            if data.get("assignees"):
                for assignee in data["assignees"]:
                    org = assignee.get("assignee_organization")
                    if org:
                        assignees.append(org)

            # Extract inventors
            inventors = []
            if data.get("inventors"):
                for inventor in data["inventors"]:
                    name = f"{inventor.get('inventor_first_name', '')} {inventor.get('inventor_last_name', '')}".strip()
                    if name:
                        inventors.append(name)

            patent_info = PatentInfo(
                patent_number=data.get("patent_number", ""),
                patent_title=data.get("patent_title", ""),
                patent_type=data.get("patent_type", ""),
                filing_date=None,  # Not directly available, needs separate lookup
                grant_date=grant_date,
                abstract=data.get("patent_abstract"),
                num_claims=data.get("patent_num_claims", 0),
                assignees=assignees,
                inventors=inventors,
                cpc_codes=[],
            )

            # Calculate expiration date
            patent_info.base_expiration_date = self.calculate_base_expiration(
                patent_info
            )
            patent_info.adjusted_expiration_date = patent_info.base_expiration_date

            return patent_info

        except Exception as e:
            logger.error(f"Error parsing patent data: {e}")
            return None

    def calculate_base_expiration(self, patent: PatentInfo) -> Optional[date]:
        """
        Calculate the base patent expiration date.

        For patents filed after June 8, 1995, the term is 20 years from filing date.
        For patents filed before that, the term is 17 years from grant date.

        Args:
            patent: PatentInfo object.

        Returns:
            Base expiration date.
        """
        # Modern patent term: 20 years from filing date
        # Since we may not have filing date, estimate from grant date
        # (Average time from filing to grant is about 2-3 years)

        if patent.filing_date:
            return patent.filing_date + relativedelta(years=self.MODERN_PATENT_TERM_YEARS)

        elif patent.grant_date:
            # Estimate filing date as 2.5 years before grant
            estimated_filing = patent.grant_date - relativedelta(months=30)
            return estimated_filing + relativedelta(years=self.MODERN_PATENT_TERM_YEARS)

        return None

    def calculate_adjusted_expiration(
        self,
        base_expiration: date,
        pta_days: int = 0,
        pte_days: int = 0,
        pediatric_extension: bool = False,
    ) -> date:
        """
        Calculate the adjusted patent expiration date including PTA, PTE, and extensions.

        Args:
            base_expiration: Base 20-year expiration date.
            pta_days: Patent Term Adjustment days (for USPTO delays).
            pte_days: Patent Term Extension days (for regulatory delays, max 5 years).
            pediatric_extension: Whether 6-month pediatric exclusivity applies.

        Returns:
            Adjusted expiration date.
        """
        adjusted = base_expiration

        # Add Patent Term Adjustment
        if pta_days > 0:
            adjusted = adjusted + timedelta(days=pta_days)

        # Add Patent Term Extension (capped at 5 years)
        if pte_days > 0:
            max_pte_days = self.MAX_PTE_YEARS * 365
            pte_days = min(pte_days, max_pte_days)
            adjusted = adjusted + timedelta(days=pte_days)

        # Add pediatric exclusivity (6 months)
        if pediatric_extension:
            adjusted = adjusted + relativedelta(months=6)

        return adjusted

    def enrich_patent_data(
        self,
        patent_number: str,
        orange_book_expiration: Optional[date] = None,
        pta_days: int = 0,
        pte_days: int = 0,
    ) -> Dict[str, Any]:
        """
        Enrich patent data with calculated expiration dates.

        Args:
            patent_number: USPTO patent number.
            orange_book_expiration: Expiration date from Orange Book (if available).
            pta_days: Patent Term Adjustment days.
            pte_days: Patent Term Extension days.

        Returns:
            Dictionary with enriched patent data.
        """
        # Get patent info from USPTO
        patent_info = self.get_patent_by_number(patent_number)

        result = {
            "patent_number": patent_number,
            "patent_title": None,
            "grant_date": None,
            "filing_date": None,
            "base_expiration_date": None,
            "pta_days": pta_days,
            "pte_days": pte_days,
            "adjusted_expiration_date": None,
            "data_source": "USPTO",
        }

        if patent_info:
            result.update(
                {
                    "patent_title": patent_info.patent_title,
                    "grant_date": patent_info.grant_date,
                    "filing_date": patent_info.filing_date,
                    "assignees": patent_info.assignees,
                    "num_claims": patent_info.num_claims,
                }
            )

        # Use Orange Book expiration if available (more accurate for pharma patents)
        if orange_book_expiration:
            result["base_expiration_date"] = orange_book_expiration
            result["adjusted_expiration_date"] = self.calculate_adjusted_expiration(
                orange_book_expiration, pta_days, pte_days
            )
        elif patent_info and patent_info.base_expiration_date:
            result["base_expiration_date"] = patent_info.base_expiration_date
            result["adjusted_expiration_date"] = self.calculate_adjusted_expiration(
                patent_info.base_expiration_date, pta_days, pte_days
            )

        return result


class PatentExpirationCalculator:
    """
    Utility class for calculating patent expiration dates.

    Handles various scenarios:
    - Standard 20-year term from filing
    - Patent Term Adjustment (PTA) for USPTO delays
    - Patent Term Extension (PTE) for FDA regulatory review delays
    - Pediatric exclusivity extensions
    """

    @staticmethod
    def calculate_expiration(
        filing_date: Optional[date] = None,
        grant_date: Optional[date] = None,
        pta_days: int = 0,
        pte_days: int = 0,
        pediatric_extension_months: int = 0,
        orange_book_expiration: Optional[date] = None,
    ) -> Dict[str, Optional[date]]:
        """
        Calculate all relevant patent expiration dates.

        Args:
            filing_date: Patent filing date.
            grant_date: Patent grant date.
            pta_days: Patent Term Adjustment days.
            pte_days: Patent Term Extension days.
            pediatric_extension_months: Pediatric exclusivity months (usually 6).
            orange_book_expiration: Pre-calculated expiration from Orange Book.

        Returns:
            Dictionary with various expiration dates.
        """
        result = {
            "base_expiration_date": None,
            "pta_adjusted_date": None,
            "pte_adjusted_date": None,
            "pediatric_exclusivity_date": None,
            "final_expiration_date": None,
        }

        # Determine base expiration
        if orange_book_expiration:
            base_expiration = orange_book_expiration
        elif filing_date:
            base_expiration = filing_date + relativedelta(years=20)
        elif grant_date:
            # Estimate filing as 2.5 years before grant
            estimated_filing = grant_date - relativedelta(months=30)
            base_expiration = estimated_filing + relativedelta(years=20)
        else:
            return result

        result["base_expiration_date"] = base_expiration

        # Add PTA
        current_expiration = base_expiration
        if pta_days > 0:
            current_expiration = current_expiration + timedelta(days=pta_days)
            result["pta_adjusted_date"] = current_expiration

        # Add PTE (max 5 years)
        if pte_days > 0:
            max_pte = 5 * 365
            pte_days = min(pte_days, max_pte)
            current_expiration = current_expiration + timedelta(days=pte_days)
            result["pte_adjusted_date"] = current_expiration

        # Add pediatric exclusivity
        if pediatric_extension_months > 0:
            current_expiration = current_expiration + relativedelta(
                months=pediatric_extension_months
            )
            result["pediatric_exclusivity_date"] = current_expiration

        result["final_expiration_date"] = current_expiration

        return result

    @staticmethod
    def days_until_expiration(expiration_date: date) -> int:
        """
        Calculate days until patent expiration.

        Args:
            expiration_date: Patent expiration date.

        Returns:
            Number of days until expiration (negative if expired).
        """
        return (expiration_date - date.today()).days

    @staticmethod
    def is_expired(expiration_date: date) -> bool:
        """
        Check if a patent has expired.

        Args:
            expiration_date: Patent expiration date.

        Returns:
            True if expired, False otherwise.
        """
        return expiration_date < date.today()

    @staticmethod
    def months_until_expiration(expiration_date: date) -> int:
        """
        Calculate months until patent expiration.

        Args:
            expiration_date: Patent expiration date.

        Returns:
            Number of months until expiration (negative if expired).
        """
        today = date.today()
        months = (expiration_date.year - today.year) * 12
        months += expiration_date.month - today.month
        return months


if __name__ == "__main__":
    # Test the USPTO extractor
    extractor = USPTOExtractor()

    # Test with a known pharmaceutical patent
    print("\n=== Testing USPTO Extractor ===")

    # Example: Humira (adalimumab) patent
    test_patent = "6090382"
    patent_info = extractor.get_patent_by_number(test_patent)

    if patent_info:
        print(f"\nPatent {test_patent}:")
        print(f"  Title: {patent_info.patent_title}")
        print(f"  Grant Date: {patent_info.grant_date}")
        print(f"  Base Expiration: {patent_info.base_expiration_date}")
        print(f"  Assignees: {', '.join(patent_info.assignees)}")
        print(f"  Claims: {patent_info.num_claims}")

    # Test expiration calculation
    print("\n=== Testing Expiration Calculation ===")
    calculator = PatentExpirationCalculator()

    test_filing_date = date(2010, 6, 15)
    expirations = calculator.calculate_expiration(
        filing_date=test_filing_date,
        pta_days=180,  # 6 months PTA
        pte_days=365,  # 1 year PTE
        pediatric_extension_months=6,
    )

    print(f"Filing Date: {test_filing_date}")
    print(f"Base Expiration: {expirations['base_expiration_date']}")
    print(f"After PTA: {expirations['pta_adjusted_date']}")
    print(f"After PTE: {expirations['pte_adjusted_date']}")
    print(f"After Pediatric: {expirations['pediatric_exclusivity_date']}")
    print(f"Final Expiration: {expirations['final_expiration_date']}")
