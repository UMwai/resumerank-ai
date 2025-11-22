"""
ClinicalTrials.gov API scraper for Clinical Trial Signal Detection System.

Uses the new ClinicalTrials.gov API v2:
https://clinicaltrials.gov/data-api/api
"""
import logging
import time
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import requests

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database.models import Trial, Company

logger = logging.getLogger(__name__)


@dataclass
class TrialSearchResult:
    """Parsed search result from ClinicalTrials.gov API."""
    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str
    conditions: List[str]
    interventions: List[str]
    enrollment: Optional[int]
    start_date: Optional[date]
    completion_date: Optional[date]
    primary_completion_date: Optional[date]
    last_update: Optional[datetime]
    raw_data: Dict[str, Any]


class ClinicalTrialsScraper:
    """Scraper for ClinicalTrials.gov API v2."""

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    def __init__(self):
        self.session = requests.Session()
        self.rate_limit = config.scraper.clinicaltrials_rate_limit
        self.timeout = config.scraper.request_timeout
        self._last_request = 0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request = time.time()

    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a rate-limited API request."""
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def search_phase3_trials(
        self,
        conditions: List[str] = None,
        sponsor_type: str = "INDUSTRY",
        min_enrollment: int = 100,
        limit: int = 20
    ) -> List[TrialSearchResult]:
        """
        Search for Phase 3 clinical trials.

        Args:
            conditions: List of conditions/indications to search
            sponsor_type: INDUSTRY, NIH, OTHER (default: INDUSTRY for biotech)
            min_enrollment: Minimum enrollment size
            limit: Maximum number of results

        Returns:
            List of TrialSearchResult objects
        """
        # Build query for Phase 3 trials
        query_parts = ["AREA[Phase](PHASE3 OR PHASE2/PHASE3)"]

        # Filter by status - active trials only
        query_parts.append("AREA[OverallStatus](RECRUITING OR ACTIVE_NOT_RECRUITING OR ENROLLING_BY_INVITATION OR NOT_YET_RECRUITING)")

        # Filter by sponsor type (industry = biotech companies)
        if sponsor_type:
            query_parts.append(f"AREA[LeadSponsorClass]({sponsor_type})")

        # Add condition filters if specified
        if conditions:
            condition_query = " OR ".join([f'AREA[Condition]("{c}")' for c in conditions])
            query_parts.append(f"({condition_query})")

        query = " AND ".join(query_parts)

        params = {
            "query.term": query,
            "pageSize": min(limit, 100),
            "sort": "LastUpdatePostDate:desc",
            "fields": ",".join([
                "NCTId", "BriefTitle", "OfficialTitle", "OverallStatus",
                "Phase", "LeadSponsorName", "LeadSponsorClass",
                "Condition", "InterventionName", "InterventionType",
                "EnrollmentCount", "EnrollmentType",
                "StartDate", "CompletionDate", "PrimaryCompletionDate",
                "LastUpdatePostDate", "StudyFirstPostDate",
                "PrimaryOutcomeMeasure", "SecondaryOutcomeMeasure",
                "StudyType", "LocationCountry", "LocationFacility"
            ])
        }

        logger.info(f"Searching ClinicalTrials.gov with query: {query[:100]}...")

        try:
            data = self._make_request("studies", params)
            studies = data.get("studies", [])
            logger.info(f"Found {len(studies)} Phase 3 trials")

            results = []
            for study in studies:
                parsed = self._parse_study(study)
                if parsed:
                    results.append(parsed)

            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search trials: {e}")
            return []

    def get_trial_details(self, nct_id: str) -> Optional[TrialSearchResult]:
        """
        Get detailed information for a specific trial.

        Args:
            nct_id: The NCT identifier (e.g., NCT12345678)

        Returns:
            TrialSearchResult with full details
        """
        try:
            data = self._make_request(f"studies/{nct_id}")
            return self._parse_study(data)
        except Exception as e:
            logger.error(f"Failed to get trial {nct_id}: {e}")
            return None

    def _parse_study(self, study_data: Dict) -> Optional[TrialSearchResult]:
        """Parse API response into TrialSearchResult."""
        try:
            # Handle both list (from search) and direct (from detail) formats
            protocol = study_data.get("protocolSection", study_data)

            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module = protocol.get("conditionsModule", {})
            interventions_module = protocol.get("armsInterventionsModule", {})
            outcomes_module = protocol.get("outcomesModule", {})

            nct_id = identification.get("nctId", "")
            if not nct_id:
                return None

            # Parse dates
            start_date = self._parse_date(status_module.get("startDateStruct", {}))
            completion_date = self._parse_date(status_module.get("completionDateStruct", {}))
            primary_completion = self._parse_date(status_module.get("primaryCompletionDateStruct", {}))
            last_update = self._parse_datetime(status_module.get("lastUpdatePostDateStruct", {}))

            # Parse sponsor
            lead_sponsor = sponsor_module.get("leadSponsor", {})
            sponsor_name = lead_sponsor.get("name", "Unknown")

            # Parse conditions
            conditions = conditions_module.get("conditions", [])

            # Parse interventions
            interventions = []
            for intervention in interventions_module.get("interventions", []):
                name = intervention.get("name", "")
                itype = intervention.get("type", "")
                if name:
                    interventions.append(f"{name} ({itype})" if itype else name)

            # Parse enrollment
            enrollment_info = design_module.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count")

            # Parse phase
            phases = design_module.get("phases", [])
            phase = phases[0] if phases else "Unknown"

            return TrialSearchResult(
                nct_id=nct_id,
                title=identification.get("briefTitle", identification.get("officialTitle", "")),
                status=status_module.get("overallStatus", "Unknown"),
                phase=phase,
                sponsor=sponsor_name,
                conditions=conditions,
                interventions=interventions,
                enrollment=enrollment,
                start_date=start_date,
                completion_date=completion_date,
                primary_completion_date=primary_completion,
                last_update=last_update,
                raw_data=study_data
            )

        except Exception as e:
            logger.warning(f"Failed to parse study: {e}")
            return None

    def _parse_date(self, date_struct: Dict) -> Optional[date]:
        """Parse date from API date structure."""
        if not date_struct:
            return None
        date_str = date_struct.get("date", "")
        if not date_str:
            return None
        try:
            # Handles formats like "2024-06-15" or "2024-06"
            if len(date_str) == 7:  # YYYY-MM format
                return datetime.strptime(date_str, "%Y-%m").date()
            return datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        except ValueError:
            return None

    def _parse_datetime(self, date_struct: Dict) -> Optional[datetime]:
        """Parse datetime from API date structure."""
        parsed_date = self._parse_date(date_struct)
        if parsed_date:
            return datetime.combine(parsed_date, datetime.min.time())
        return None

    def fetch_and_store_trials(
        self,
        conditions: List[str] = None,
        limit: int = 20,
        company_tickers: Dict[str, str] = None
    ) -> Tuple[int, int]:
        """
        Fetch trials and store them in the database.

        Args:
            conditions: List of conditions to search
            limit: Maximum number of trials to fetch
            company_tickers: Optional mapping of sponsor names to tickers

        Returns:
            Tuple of (new_trials_count, updated_trials_count)
        """
        # Default priority conditions from spec
        if conditions is None:
            conditions = config.monitoring.priority_indications

        trials = self.search_phase3_trials(
            conditions=conditions,
            limit=limit
        )

        new_count = 0
        updated_count = 0

        for trial_result in trials:
            # Check if trial already exists
            existing = Trial.get_by_id(trial_result.nct_id)

            # Try to match sponsor to company ticker
            ticker = None
            if company_tickers:
                sponsor_lower = trial_result.sponsor.lower()
                for company_name, company_ticker in company_tickers.items():
                    if company_name.lower() in sponsor_lower or sponsor_lower in company_name.lower():
                        ticker = company_ticker
                        break

            trial = Trial(
                trial_id=trial_result.nct_id,
                company_ticker=ticker,
                drug_name=trial_result.interventions[0] if trial_result.interventions else None,
                indication=", ".join(trial_result.conditions[:3]),  # First 3 conditions
                phase=trial_result.phase,
                enrollment_target=trial_result.enrollment,
                enrollment_current=None,  # Would need additional API call
                start_date=trial_result.start_date,
                expected_completion=trial_result.completion_date,
                primary_completion_date=trial_result.primary_completion_date,
                status=trial_result.status,
                sponsor=trial_result.sponsor,
                study_type="INTERVENTIONAL",
                last_updated=trial_result.last_update or datetime.now(),
                raw_data=trial_result.raw_data
            )

            trial.save()

            if existing:
                updated_count += 1
            else:
                new_count += 1

        logger.info(f"Stored {new_count} new trials, updated {updated_count} existing trials")
        return new_count, updated_count

    def refresh_trial(self, nct_id: str) -> Optional[Trial]:
        """
        Refresh a single trial's data from the API.

        Args:
            nct_id: The NCT identifier

        Returns:
            Updated Trial object or None if failed
        """
        trial_result = self.get_trial_details(nct_id)
        if not trial_result:
            return None

        existing = Trial.get_by_id(nct_id)

        trial = Trial(
            trial_id=trial_result.nct_id,
            company_ticker=existing.company_ticker if existing else None,
            drug_name=trial_result.interventions[0] if trial_result.interventions else None,
            indication=", ".join(trial_result.conditions[:3]),
            phase=trial_result.phase,
            enrollment_target=trial_result.enrollment,
            enrollment_current=existing.enrollment_current if existing else None,
            start_date=trial_result.start_date,
            expected_completion=trial_result.completion_date,
            primary_completion_date=trial_result.primary_completion_date,
            status=trial_result.status,
            sponsor=trial_result.sponsor,
            study_type="INTERVENTIONAL",
            last_updated=trial_result.last_update or datetime.now(),
            raw_data=trial_result.raw_data
        )

        trial.save()
        logger.info(f"Refreshed trial {nct_id}")
        return trial


# Mapping of known biotech companies to their sponsors on ClinicalTrials.gov
BIOTECH_SPONSOR_MAPPING = {
    "Cassava Sciences": "SAVA",
    "Moderna": "MRNA",
    "Novavax": "NVAX",
    "Ionis Pharmaceuticals": "IONS",
    "Alnylam Pharmaceuticals": "ALNY",
    "BioMarin": "BMRN",
    "Sarepta Therapeutics": "SRPT",
    "Ultragenyx": "RARE",
    "Neurocrine Biosciences": "NBIX",
    "ACADIA Pharmaceuticals": "ACAD",
    # Add more mappings as needed
}


if __name__ == "__main__":
    # Test the scraper
    import sys
    logging.basicConfig(level=logging.INFO)

    scraper = ClinicalTrialsScraper()

    print("Searching for Phase 3 oncology trials...")
    results = scraper.search_phase3_trials(
        conditions=["cancer", "oncology"],
        limit=5
    )

    for trial in results:
        print(f"\n{trial.nct_id}: {trial.title[:60]}...")
        print(f"  Status: {trial.status}")
        print(f"  Phase: {trial.phase}")
        print(f"  Sponsor: {trial.sponsor}")
        print(f"  Conditions: {', '.join(trial.conditions[:2])}")
        print(f"  Enrollment: {trial.enrollment}")
