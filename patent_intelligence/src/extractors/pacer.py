"""
PACER Court Filing Extractor

Extracts patent litigation data from PACER (Public Access to Court Electronic Records).
PACER provides access to federal court filings, essential for tracking pharmaceutical
patent litigation and Paragraph IV challenges.

Data source: https://pacer.uscourts.gov/
"""

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.logger import get_logger
from ..utils.data_validation import (
    DataValidator,
    ValidationResult,
    validate_required_fields,
    validate_date_range,
)

logger = get_logger(__name__)


@dataclass
class PACERCredentials:
    """PACER login credentials."""
    username: str
    password: str
    client_code: Optional[str] = None  # For billing purposes


@dataclass
class CourtFiling:
    """Data class for a court filing record."""
    case_id: str
    case_number: str
    case_name: str
    court: str
    court_type: str  # DISTRICT, APPEALS, SUPREME
    jurisdiction: str
    filing_date: date
    case_type: str  # PATENT, ANDA, IPR, PGR
    nature_of_suit: str
    cause_of_action: str
    plaintiff: str
    defendant: str
    plaintiff_counsel: Optional[str] = None
    defendant_counsel: Optional[str] = None
    judge_assigned: Optional[str] = None
    status: str = "OPEN"  # OPEN, CLOSED, STAYED
    docket_entries: List[Dict] = field(default_factory=list)
    related_patents: List[str] = field(default_factory=list)
    related_drug_names: List[str] = field(default_factory=list)
    pacer_link: Optional[str] = None
    last_activity_date: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "case_id": self.case_id,
            "case_number": self.case_number,
            "case_name": self.case_name,
            "court": self.court,
            "court_type": self.court_type,
            "jurisdiction": self.jurisdiction,
            "filing_date": self.filing_date,
            "case_type": self.case_type,
            "nature_of_suit": self.nature_of_suit,
            "cause_of_action": self.cause_of_action,
            "plaintiff": self.plaintiff,
            "defendant": self.defendant,
            "plaintiff_counsel": self.plaintiff_counsel,
            "defendant_counsel": self.defendant_counsel,
            "judge_assigned": self.judge_assigned,
            "status": self.status,
            "related_patents": self.related_patents,
            "related_drug_names": self.related_drug_names,
            "pacer_link": self.pacer_link,
            "last_activity_date": self.last_activity_date,
        }


@dataclass
class DocketEntry:
    """Data class for a docket entry."""
    entry_number: int
    filing_date: date
    description: str
    filed_by: Optional[str] = None
    document_number: Optional[str] = None
    attachment_count: int = 0
    is_sealed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_number": self.entry_number,
            "filing_date": self.filing_date,
            "description": self.description,
            "filed_by": self.filed_by,
            "document_number": self.document_number,
            "attachment_count": self.attachment_count,
            "is_sealed": self.is_sealed,
        }


class PACERExtractor:
    """
    Extracts patent litigation data from PACER.

    PACER is the official electronic records system for federal courts.
    This extractor focuses on:
    - Paragraph IV patent challenges (Hatch-Waxman)
    - Inter Partes Review (IPR) at PTAB
    - Post Grant Review (PGR)
    - Standard patent infringement cases

    Note: PACER charges $0.10 per page with a $3.00 cap per document.
    This extractor implements caching to minimize costs.
    """

    # PACER API endpoints
    PACER_LOGIN_URL = "https://pacer.login.uscourts.gov/csologin/login.jsf"
    PACER_SEARCH_URL = "https://pcl.uscourts.gov/pcl/search.jsf"
    PACER_CASE_LOCATOR = "https://pcl.uscourts.gov/pcl/pages/search/findCase.jsf"

    # Court codes for pharmaceutical cases (commonly used districts)
    PHARMA_COURTS = {
        "deb": "District of Delaware",  # Most common for pharma
        "njd": "District of New Jersey",
        "txed": "Eastern District of Texas",
        "nysd": "Southern District of New York",
        "ilnd": "Northern District of Illinois",
        "cand": "Northern District of California",
        "cacd": "Central District of California",
        "masd": "District of Massachusetts",
        "paed": "Eastern District of Pennsylvania",
        "flsd": "Southern District of Florida",
    }

    # Nature of suit codes relevant to pharmaceutical patents
    PHARMA_NOS_CODES = {
        "830": "Patent",
        "835": "Patent - ANDA Certification",
        "840": "Trademark",
        "893": "Administrative Procedures Act",
    }

    # Key terms for pharmaceutical patent litigation
    PHARMA_KEYWORDS = [
        "ANDA", "paragraph IV", "Hatch-Waxman",
        "patent infringement", "505(b)(2)",
        "generic drug", "biologic", "biosimilar",
        "Orange Book", "Purple Book",
    ]

    def __init__(
        self,
        credentials: Optional[PACERCredentials] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
        monthly_budget: float = 30.0,  # USD
    ):
        """
        Initialize the PACER extractor.

        Args:
            credentials: PACER login credentials.
            cache_dir: Directory for caching responses.
            use_cache: Whether to use cached data.
            cache_ttl_hours: Cache time-to-live in hours.
            monthly_budget: Monthly spending limit on PACER.
        """
        self.credentials = credentials
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/pacer")
        self.use_cache = use_cache
        self.cache_ttl_hours = cache_ttl_hours
        self.monthly_budget = monthly_budget

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session management
        self._session: Optional[requests.Session] = None
        self._is_authenticated = False

        # Cost tracking
        self._current_month_cost = 0.0
        self._cost_file = self.cache_dir / "monthly_cost.txt"
        self._load_monthly_cost()

        # Data validator
        self._validator = DataValidator()

    def _load_monthly_cost(self) -> None:
        """Load the current month's spending from cache."""
        try:
            if self._cost_file.exists():
                content = self._cost_file.read_text().strip()
                month, cost = content.split(",")
                if month == datetime.now().strftime("%Y-%m"):
                    self._current_month_cost = float(cost)
                else:
                    self._current_month_cost = 0.0
        except (ValueError, IOError):
            self._current_month_cost = 0.0

    def _save_monthly_cost(self) -> None:
        """Save the current month's spending to cache."""
        month = datetime.now().strftime("%Y-%m")
        self._cost_file.write_text(f"{month},{self._current_month_cost:.2f}")

    def _check_budget(self, estimated_cost: float) -> bool:
        """
        Check if the estimated cost is within budget.

        Args:
            estimated_cost: Estimated cost in USD.

        Returns:
            True if within budget, False otherwise.
        """
        if self._current_month_cost + estimated_cost > self.monthly_budget:
            logger.warning(
                f"PACER budget exceeded: ${self._current_month_cost:.2f} spent, "
                f"${self.monthly_budget:.2f} limit"
            )
            return False
        return True

    def _add_cost(self, cost: float) -> None:
        """Add cost to monthly tracking."""
        self._current_month_cost += cost
        self._save_monthly_cost()
        logger.debug(f"PACER cost: ${cost:.2f}, Month total: ${self._current_month_cost:.2f}")

    def _get_session(self) -> requests.Session:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "User-Agent": "Mozilla/5.0 (compatible; PatentIntelligence/1.0)",
                "Accept": "text/html,application/json",
            })
        return self._session

    def _get_cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        return hashlib.md5(query.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[Dict]:
        """Get cached response if valid."""
        if not self.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check cache age
        file_age_hours = (
            datetime.now().timestamp() - cache_file.stat().st_mtime
        ) / 3600

        if file_age_hours > self.cache_ttl_hours:
            return None

        try:
            import json
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, IOError):
            return None

    def _cache_response(self, cache_key: str, data: Dict) -> None:
        """Cache response data."""
        import json
        cache_file = self.cache_dir / f"{cache_key}.json"
        cache_file.write_text(json.dumps(data))

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def authenticate(self) -> bool:
        """
        Authenticate with PACER.

        Returns:
            True if authentication successful.
        """
        if not self.credentials:
            logger.warning("No PACER credentials provided - using demo mode")
            return False

        logger.info("Authenticating with PACER...")

        session = self._get_session()

        try:
            # Get login page for CSRF token
            response = session.get(self.PACER_LOGIN_URL, timeout=30)
            response.raise_for_status()

            # Extract CSRF token from page
            soup = BeautifulSoup(response.text, "lxml")
            csrf_token = soup.find("input", {"name": "javax.faces.ViewState"})

            if csrf_token:
                csrf_value = csrf_token.get("value", "")
            else:
                csrf_value = ""

            # Submit login form
            login_data = {
                "loginForm:username": self.credentials.username,
                "loginForm:password": self.credentials.password,
                "loginForm:clientCode": self.credentials.client_code or "",
                "javax.faces.ViewState": csrf_value,
                "loginForm:submitLogin": "Login",
            }

            response = session.post(
                self.PACER_LOGIN_URL,
                data=login_data,
                timeout=30,
            )

            # Check for successful login
            if "Welcome" in response.text or "logged in" in response.text.lower():
                self._is_authenticated = True
                logger.info("PACER authentication successful")
                return True
            else:
                logger.error("PACER authentication failed")
                return False

        except requests.RequestException as e:
            logger.error(f"PACER authentication error: {e}")
            return False

    def search_cases(
        self,
        party_name: Optional[str] = None,
        patent_number: Optional[str] = None,
        drug_name: Optional[str] = None,
        case_type: Optional[str] = None,
        court: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        nature_of_suit: Optional[str] = "830",  # Patent cases
        limit: int = 100,
    ) -> List[CourtFiling]:
        """
        Search for court cases matching criteria.

        Args:
            party_name: Party name to search (plaintiff or defendant).
            patent_number: Patent number involved.
            drug_name: Drug name involved.
            case_type: Type of case (PATENT, ANDA, IPR, PGR).
            court: Court code (e.g., "deb" for Delaware).
            date_from: Start date for search.
            date_to: End date for search.
            nature_of_suit: PACER nature of suit code.
            limit: Maximum number of results.

        Returns:
            List of CourtFiling objects.
        """
        # Build search query for cache key
        query_parts = []
        if party_name:
            query_parts.append(f"party:{party_name}")
        if patent_number:
            query_parts.append(f"patent:{patent_number}")
        if drug_name:
            query_parts.append(f"drug:{drug_name}")
        if court:
            query_parts.append(f"court:{court}")
        if date_from:
            query_parts.append(f"from:{date_from}")
        if date_to:
            query_parts.append(f"to:{date_to}")

        cache_key = self._get_cache_key("|".join(query_parts))

        # Check cache
        cached = self._get_cached_response(cache_key)
        if cached:
            logger.info(f"Using cached PACER results for: {query_parts}")
            return [CourtFiling(**filing) for filing in cached.get("filings", [])]

        # Check budget
        estimated_cost = 0.30  # Estimate per search
        if not self._check_budget(estimated_cost):
            logger.warning("PACER budget exceeded, using demo data")
            return self._get_demo_filings(party_name, drug_name, limit)

        # If not authenticated, use demo mode
        if not self._is_authenticated:
            logger.info("PACER not authenticated, using demo data")
            return self._get_demo_filings(party_name, drug_name, limit)

        # Execute actual PACER search
        try:
            filings = self._execute_pacer_search(
                party_name=party_name,
                patent_number=patent_number,
                drug_name=drug_name,
                case_type=case_type,
                court=court,
                date_from=date_from,
                date_to=date_to,
                nature_of_suit=nature_of_suit,
                limit=limit,
            )

            # Cache results
            self._cache_response(cache_key, {
                "filings": [f.to_dict() for f in filings],
                "query": query_parts,
                "timestamp": datetime.now().isoformat(),
            })

            # Track cost
            self._add_cost(estimated_cost)

            return filings

        except Exception as e:
            logger.error(f"PACER search failed: {e}")
            return self._get_demo_filings(party_name, drug_name, limit)

    def _execute_pacer_search(
        self,
        party_name: Optional[str] = None,
        patent_number: Optional[str] = None,
        drug_name: Optional[str] = None,
        case_type: Optional[str] = None,
        court: Optional[str] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        nature_of_suit: Optional[str] = None,
        limit: int = 100,
    ) -> List[CourtFiling]:
        """Execute actual PACER search (requires authentication)."""
        session = self._get_session()

        # Build search parameters
        search_params = {
            "pageSize": min(limit, 100),
            "natureOfSuit": nature_of_suit or "830",
        }

        if party_name:
            search_params["partyName"] = party_name
        if court:
            search_params["courtCode"] = court
        if date_from:
            search_params["filedFrom"] = date_from.strftime("%m/%d/%Y")
        if date_to:
            search_params["filedTo"] = date_to.strftime("%m/%d/%Y")

        try:
            response = session.post(
                self.PACER_CASE_LOCATOR,
                data=search_params,
                timeout=60,
            )
            response.raise_for_status()

            # Parse results
            return self._parse_search_results(response.text, drug_name, patent_number)

        except requests.RequestException as e:
            logger.error(f"PACER search request failed: {e}")
            raise

    def _parse_search_results(
        self,
        html: str,
        drug_filter: Optional[str] = None,
        patent_filter: Optional[str] = None,
    ) -> List[CourtFiling]:
        """Parse PACER search results HTML."""
        soup = BeautifulSoup(html, "lxml")
        filings = []

        # Find results table (structure varies by PACER version)
        results_table = soup.find("table", {"id": "results"}) or soup.find("table", class_="results")

        if not results_table:
            logger.warning("No results table found in PACER response")
            return filings

        rows = results_table.find_all("tr")[1:]  # Skip header

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 5:
                continue

            try:
                case_link = cells[0].find("a")
                case_number = case_link.text.strip() if case_link else cells[0].text.strip()
                case_name = cells[1].text.strip()
                court = cells[2].text.strip()
                filing_date_str = cells[3].text.strip()
                nature_of_suit = cells[4].text.strip()

                # Parse filing date
                try:
                    filing_date = datetime.strptime(filing_date_str, "%m/%d/%Y").date()
                except ValueError:
                    filing_date = date.today()

                # Extract parties from case name
                plaintiff, defendant = self._extract_parties(case_name)

                # Generate case ID
                case_id = hashlib.md5(
                    f"{case_number}|{court}|{filing_date}".encode()
                ).hexdigest()[:16]

                # Determine case type
                case_type = self._determine_case_type(case_name, nature_of_suit)

                # Extract patent numbers from case name
                related_patents = self._extract_patent_numbers(case_name)

                # Check filters
                if drug_filter and drug_filter.lower() not in case_name.lower():
                    continue
                if patent_filter and patent_filter not in str(related_patents):
                    continue

                filing = CourtFiling(
                    case_id=case_id,
                    case_number=case_number,
                    case_name=case_name,
                    court=court,
                    court_type="DISTRICT",
                    jurisdiction=self._get_jurisdiction(court),
                    filing_date=filing_date,
                    case_type=case_type,
                    nature_of_suit=nature_of_suit,
                    cause_of_action="Patent Infringement",
                    plaintiff=plaintiff,
                    defendant=defendant,
                    related_patents=related_patents,
                    pacer_link=case_link.get("href") if case_link else None,
                )

                filings.append(filing)

            except Exception as e:
                logger.warning(f"Error parsing PACER row: {e}")
                continue

        logger.info(f"Parsed {len(filings)} filings from PACER results")
        return filings

    def _extract_parties(self, case_name: str) -> Tuple[str, str]:
        """Extract plaintiff and defendant from case name."""
        # Common formats: "Plaintiff v. Defendant", "Plaintiff vs. Defendant"
        patterns = [
            r"(.+?)\s+v\.?\s+(.+)",
            r"(.+?)\s+vs\.?\s+(.+)",
        ]

        for pattern in patterns:
            match = re.match(pattern, case_name, re.IGNORECASE)
            if match:
                return match.group(1).strip(), match.group(2).strip()

        return case_name, "Unknown"

    def _extract_patent_numbers(self, text: str) -> List[str]:
        """Extract patent numbers from text."""
        # Patent number patterns: US1234567, 1,234,567, RE12345
        patterns = [
            r"US\s*(\d{7,8})",
            r"(\d{1,2},\d{3},\d{3})",
            r"RE(\d{5})",
            r"patent[^\d]*(\d{7,8})",
        ]

        patents = set()
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize patent number
                patent_no = re.sub(r"[,\s]", "", match)
                patents.add(patent_no)

        return list(patents)

    def _determine_case_type(self, case_name: str, nature_of_suit: str) -> str:
        """Determine the type of patent case."""
        case_name_lower = case_name.lower()

        if "anda" in case_name_lower or "paragraph iv" in case_name_lower:
            return "ANDA"
        elif "ipr" in case_name_lower or "inter partes" in case_name_lower:
            return "IPR"
        elif "pgr" in case_name_lower or "post grant" in case_name_lower:
            return "PGR"
        elif nature_of_suit == "835":
            return "ANDA"
        else:
            return "PATENT"

    def _get_jurisdiction(self, court: str) -> str:
        """Get jurisdiction from court code."""
        court_lower = court.lower()

        # Check known courts
        for code, name in self.PHARMA_COURTS.items():
            if code in court_lower or name.lower() in court_lower:
                return name

        return court

    def _get_demo_filings(
        self,
        party_name: Optional[str] = None,
        drug_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[CourtFiling]:
        """
        Return demo filings for testing without PACER access.

        These are based on real historical pharmaceutical patent cases.
        """
        demo_cases = [
            {
                "case_id": "demo_humira_001",
                "case_number": "1:17-cv-01065",
                "case_name": "AbbVie Inc. v. Sandoz Inc.",
                "court": "District of Delaware",
                "court_type": "DISTRICT",
                "jurisdiction": "District of Delaware",
                "filing_date": date(2023, 3, 15),
                "case_type": "ANDA",
                "nature_of_suit": "835",
                "cause_of_action": "Patent Infringement - ANDA",
                "plaintiff": "AbbVie Inc.",
                "defendant": "Sandoz Inc.",
                "status": "CLOSED",
                "related_patents": ["9017680", "9359434"],
                "related_drug_names": ["Humira", "adalimumab"],
            },
            {
                "case_id": "demo_keytruda_001",
                "case_number": "1:22-cv-00892",
                "case_name": "Merck Sharp & Dohme Corp. v. Teva Pharmaceuticals",
                "court": "District of New Jersey",
                "court_type": "DISTRICT",
                "jurisdiction": "District of New Jersey",
                "filing_date": date(2023, 6, 20),
                "case_type": "PATENT",
                "nature_of_suit": "830",
                "cause_of_action": "Patent Infringement",
                "plaintiff": "Merck Sharp & Dohme Corp.",
                "defendant": "Teva Pharmaceuticals",
                "status": "OPEN",
                "related_patents": ["8354509", "8952136"],
                "related_drug_names": ["Keytruda", "pembrolizumab"],
            },
            {
                "case_id": "demo_eliquis_001",
                "case_number": "1:20-cv-01376",
                "case_name": "Bristol-Myers Squibb v. Aurobindo Pharma",
                "court": "District of Delaware",
                "court_type": "DISTRICT",
                "jurisdiction": "District of Delaware",
                "filing_date": date(2023, 2, 10),
                "case_type": "ANDA",
                "nature_of_suit": "835",
                "cause_of_action": "Patent Infringement - ANDA",
                "plaintiff": "Bristol-Myers Squibb",
                "defendant": "Aurobindo Pharma",
                "status": "CLOSED",
                "related_patents": ["9326945", "10052335"],
                "related_drug_names": ["Eliquis", "apixaban"],
            },
            {
                "case_id": "demo_stelara_001",
                "case_number": "3:21-cv-01932",
                "case_name": "Janssen Biotech v. Amgen Inc.",
                "court": "District of New Jersey",
                "court_type": "DISTRICT",
                "jurisdiction": "District of New Jersey",
                "filing_date": date(2023, 8, 5),
                "case_type": "PATENT",
                "nature_of_suit": "830",
                "cause_of_action": "Patent Infringement",
                "plaintiff": "Janssen Biotech",
                "defendant": "Amgen Inc.",
                "status": "OPEN",
                "related_patents": ["6902734", "7405065"],
                "related_drug_names": ["Stelara", "ustekinumab"],
            },
            {
                "case_id": "demo_revlimid_001",
                "case_number": "1:19-cv-01023",
                "case_name": "Celgene Corp. v. Natco Pharma Ltd.",
                "court": "District of Delaware",
                "court_type": "DISTRICT",
                "jurisdiction": "District of Delaware",
                "filing_date": date(2022, 9, 12),
                "case_type": "ANDA",
                "nature_of_suit": "835",
                "cause_of_action": "Patent Infringement - ANDA",
                "plaintiff": "Celgene Corp.",
                "defendant": "Natco Pharma Ltd.",
                "status": "CLOSED",
                "related_patents": ["5635517", "7855217"],
                "related_drug_names": ["Revlimid", "lenalidomide"],
            },
        ]

        # Filter by party name or drug name if provided
        filtered_cases = []
        for case in demo_cases:
            if party_name:
                if party_name.lower() not in case["plaintiff"].lower() and \
                   party_name.lower() not in case["defendant"].lower():
                    continue
            if drug_name:
                drug_names_lower = [d.lower() for d in case["related_drug_names"]]
                if drug_name.lower() not in drug_names_lower and \
                   not any(drug_name.lower() in d for d in drug_names_lower):
                    continue
            filtered_cases.append(case)

        # Convert to CourtFiling objects
        filings = []
        for case in filtered_cases[:limit]:
            filings.append(CourtFiling(
                case_id=case["case_id"],
                case_number=case["case_number"],
                case_name=case["case_name"],
                court=case["court"],
                court_type=case["court_type"],
                jurisdiction=case["jurisdiction"],
                filing_date=case["filing_date"],
                case_type=case["case_type"],
                nature_of_suit=case["nature_of_suit"],
                cause_of_action=case["cause_of_action"],
                plaintiff=case["plaintiff"],
                defendant=case["defendant"],
                status=case["status"],
                related_patents=case["related_patents"],
                related_drug_names=case["related_drug_names"],
            ))

        logger.info(f"Returning {len(filings)} demo PACER filings")
        return filings

    def get_case_docket(self, case_id: str, court: str) -> List[DocketEntry]:
        """
        Get docket entries for a specific case.

        Args:
            case_id: PACER case ID.
            court: Court code.

        Returns:
            List of DocketEntry objects.
        """
        # Check budget
        estimated_cost = 0.50  # Per docket request
        if not self._check_budget(estimated_cost):
            logger.warning("Budget exceeded for docket request")
            return []

        cache_key = self._get_cache_key(f"docket:{case_id}:{court}")

        # Check cache
        cached = self._get_cached_response(cache_key)
        if cached:
            return [DocketEntry(**entry) for entry in cached.get("entries", [])]

        # If not authenticated, return demo data
        if not self._is_authenticated:
            return self._get_demo_docket_entries()

        # Execute actual docket fetch
        try:
            session = self._get_session()

            response = session.get(
                f"{self.PACER_CASE_LOCATOR}/case/{case_id}/docket",
                params={"court": court},
                timeout=60,
            )
            response.raise_for_status()

            entries = self._parse_docket_entries(response.text)

            # Cache results
            self._cache_response(cache_key, {
                "entries": [e.to_dict() for e in entries],
                "timestamp": datetime.now().isoformat(),
            })

            self._add_cost(estimated_cost)

            return entries

        except Exception as e:
            logger.error(f"Docket fetch failed: {e}")
            return self._get_demo_docket_entries()

    def _parse_docket_entries(self, html: str) -> List[DocketEntry]:
        """Parse docket entries from HTML."""
        soup = BeautifulSoup(html, "lxml")
        entries = []

        docket_table = soup.find("table", {"id": "docket"}) or soup.find("table", class_="docket")

        if not docket_table:
            return entries

        rows = docket_table.find_all("tr")[1:]

        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            try:
                entry_num = int(cells[0].text.strip())
                date_str = cells[1].text.strip()
                description = cells[2].text.strip()

                filing_date = datetime.strptime(date_str, "%m/%d/%Y").date()

                entries.append(DocketEntry(
                    entry_number=entry_num,
                    filing_date=filing_date,
                    description=description,
                ))

            except (ValueError, IndexError) as e:
                logger.debug(f"Error parsing docket entry: {e}")
                continue

        return entries

    def _get_demo_docket_entries(self) -> List[DocketEntry]:
        """Return demo docket entries."""
        return [
            DocketEntry(
                entry_number=1,
                filing_date=date(2023, 1, 15),
                description="COMPLAINT for Patent Infringement filed by Plaintiff",
                filed_by="Plaintiff",
            ),
            DocketEntry(
                entry_number=5,
                filing_date=date(2023, 2, 20),
                description="ANSWER to Complaint filed by Defendant",
                filed_by="Defendant",
            ),
            DocketEntry(
                entry_number=12,
                filing_date=date(2023, 5, 10),
                description="MOTION for Summary Judgment filed by Plaintiff",
                filed_by="Plaintiff",
            ),
            DocketEntry(
                entry_number=25,
                filing_date=date(2023, 8, 1),
                description="ORDER granting in part Motion for Summary Judgment",
                filed_by="Court",
            ),
        ]

    def search_pharma_litigation(
        self,
        drug_names: List[str],
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
        limit: int = 100,
    ) -> List[CourtFiling]:
        """
        Search for pharmaceutical patent litigation by drug names.

        Args:
            drug_names: List of drug names to search.
            date_from: Start date.
            date_to: End date.
            limit: Maximum results.

        Returns:
            List of CourtFiling objects.
        """
        all_filings = []
        seen_case_ids = set()

        # Default date range: last 3 years
        if not date_from:
            date_from = date.today() - timedelta(days=365 * 3)
        if not date_to:
            date_to = date.today()

        for drug_name in drug_names:
            filings = self.search_cases(
                drug_name=drug_name,
                date_from=date_from,
                date_to=date_to,
                limit=limit // len(drug_names) + 1,
            )

            for filing in filings:
                if filing.case_id not in seen_case_ids:
                    seen_case_ids.add(filing.case_id)
                    all_filings.append(filing)

        logger.info(f"Found {len(all_filings)} unique pharma litigation cases")
        return all_filings[:limit]

    def extract_for_database(
        self,
        drug_names: Optional[List[str]] = None,
        patent_numbers: Optional[List[str]] = None,
        date_from: Optional[date] = None,
        date_to: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract litigation data formatted for database loading.

        Args:
            drug_names: Drug names to search.
            patent_numbers: Patent numbers to search.
            date_from: Start date.
            date_to: End date.

        Returns:
            List of litigation records for database.
        """
        filings = []

        if drug_names:
            filings.extend(self.search_pharma_litigation(
                drug_names=drug_names,
                date_from=date_from,
                date_to=date_to,
            ))

        if patent_numbers:
            for patent_no in patent_numbers:
                filings.extend(self.search_cases(
                    patent_number=patent_no,
                    date_from=date_from,
                    date_to=date_to,
                ))

        # Deduplicate
        seen = set()
        unique_filings = []
        for filing in filings:
            if filing.case_id not in seen:
                seen.add(filing.case_id)
                unique_filings.append(filing)

        # Convert to database format
        db_records = []
        for filing in unique_filings:
            for patent_no in filing.related_patents or [None]:
                record = {
                    "case_id": filing.case_id,
                    "case_name": filing.case_name,
                    "patent_number": patent_no,
                    "plaintiff": filing.plaintiff,
                    "defendant": filing.defendant,
                    "court": filing.court,
                    "jurisdiction": filing.jurisdiction,
                    "case_type": filing.case_type,
                    "filing_date": filing.filing_date,
                    "outcome": "ONGOING" if filing.status == "OPEN" else filing.status,
                    "pacer_link": filing.pacer_link,
                    "data_source": "PACER",
                }

                # Validate record
                validation = self._validator.validate_litigation(record)
                if validation.is_valid:
                    db_records.append(record)
                else:
                    logger.warning(f"Invalid litigation record: {validation.errors}")

        logger.info(f"Extracted {len(db_records)} litigation records for database")
        return db_records

    def close(self) -> None:
        """Close the extractor and save state."""
        self._save_monthly_cost()
        if self._session:
            self._session.close()
            self._session = None
        logger.info("PACER extractor closed")


# Convenience function for creating extractor
def get_pacer_extractor(
    username: Optional[str] = None,
    password: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> PACERExtractor:
    """
    Create a PACER extractor with optional credentials.

    Args:
        username: PACER username.
        password: PACER password.
        cache_dir: Cache directory.

    Returns:
        PACERExtractor instance.
    """
    credentials = None
    if username and password:
        credentials = PACERCredentials(username=username, password=password)

    return PACERExtractor(
        credentials=credentials,
        cache_dir=cache_dir,
    )


if __name__ == "__main__":
    # Test the extractor in demo mode
    extractor = PACERExtractor()

    print("\n=== Testing PACER Extractor (Demo Mode) ===")

    # Search for Humira litigation
    filings = extractor.search_cases(drug_name="Humira")
    print(f"\nFound {len(filings)} Humira-related filings:")
    for filing in filings:
        print(f"  - {filing.case_name}")
        print(f"    Court: {filing.court}, Filed: {filing.filing_date}")
        print(f"    Status: {filing.status}, Type: {filing.case_type}")

    # Get database-ready records
    db_records = extractor.extract_for_database(
        drug_names=["Humira", "Keytruda", "Eliquis"]
    )
    print(f"\nExtracted {len(db_records)} records for database")

    extractor.close()
