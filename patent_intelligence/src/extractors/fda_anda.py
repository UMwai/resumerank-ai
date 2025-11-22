"""
FDA ANDA (Abbreviated New Drug Application) Extractor

Extracts generic drug application data from FDA databases.
Tracks ANDA filings, approvals, and first-to-file status.

Data sources:
- FDA Drugs@FDA database
- FDA Approved Drug Products with Therapeutic Equivalence Evaluations (Orange Book)
- FDA Generic Drug Approvals Reports
"""

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ANDAApplication:
    """Data class for ANDA application information."""

    anda_number: str
    drug_name: str
    active_ingredient: str
    dosage_form: str
    strength: str
    applicant_company: str
    approval_date: Optional[date]
    tentative_approval_date: Optional[date]
    first_to_file: bool
    paragraph_iv: bool
    status: str  # PENDING, TENTATIVE, APPROVED, WITHDRAWN
    reference_nda: Optional[str]  # Reference Listed Drug NDA number


class ANDAExtractor:
    """
    Extracts ANDA (generic drug application) data from FDA sources.

    The ANDA process allows generic manufacturers to get approval for
    generic versions of already approved drugs by demonstrating
    bioequivalence rather than conducting full clinical trials.
    """

    # FDA Drugs@FDA search URL
    DRUGS_FDA_URL = "https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm"

    # FDA Generic Drug Approvals
    GENERIC_APPROVALS_URL = (
        "https://www.fda.gov/drugs/drug-and-biologic-approval-and-ind-activity-reports/"
        "anda-approvals"
    )

    # Major generic pharmaceutical companies
    GENERIC_COMPANIES = {
        "TEVA": "TEVA",
        "MYLAN": "VTRS",  # Now Viatris
        "VIATRIS": "VTRS",
        "SANDOZ": None,  # Part of Novartis
        "DR. REDDY": "RDY",
        "SUN PHARMA": "SUNPHARMA",
        "LUPIN": "LUPIN",
        "AUROBINDO": "AUROPHARMA",
        "APOTEX": None,  # Private
        "AMNEAL": "AMRX",
        "ZYDUS": "ZYDUSLIFE",
        "HIKMA": "HIK",
        "CIPLA": "CIPLA",
        "GLENMARK": "GLENMARK",
        "TORRENT": "TORNTPHARM",
        "FRESENIUS": "FRE",
        "PERRIGO": "PRGO",
        "IMPAX": None,  # Acquired
        "ACTAVIS": None,  # Acquired by Teva
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the ANDA extractor.

        Args:
            cache_dir: Directory for caching downloaded files.
        """
        from pathlib import Path

        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/anda")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load Orange Book data for ANDA information
        self._orange_book_products: Optional[pd.DataFrame] = None

    def _load_orange_book_anda(self) -> pd.DataFrame:
        """
        Load ANDA records from Orange Book data.

        Returns:
            DataFrame of ANDA products.
        """
        from .orange_book import OrangeBookExtractor

        extractor = OrangeBookExtractor(cache_dir=str(self.cache_dir / "orange_book"))
        extractor.fetch_data()

        # Get all products
        if extractor._products_df is not None:
            # Filter for ANDAs (generic applications)
            anda_df = extractor._products_df[
                extractor._products_df["APPL_TYPE"] == "A"
            ].copy()

            logger.info(f"Loaded {len(anda_df)} ANDA records from Orange Book")
            return anda_df

        return pd.DataFrame()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _fetch_page(self, url: str) -> str:
        """
        Fetch a web page with retry logic.

        Args:
            url: URL to fetch.

        Returns:
            Page content as string.
        """
        response = requests.get(
            url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; PatentIntelligence/1.0)"
            },
        )
        response.raise_for_status()
        return response.text

    def get_andas_for_nda(self, nda_number: str) -> List[ANDAApplication]:
        """
        Get all ANDA applications referencing a specific NDA.

        Args:
            nda_number: NDA number of the reference listed drug.

        Returns:
            List of ANDA applications.
        """
        if self._orange_book_products is None:
            self._orange_book_products = self._load_orange_book_anda()

        # Find ANDAs that reference this NDA (via trade name/ingredient matching)
        # Note: Direct NDA-ANDA linkage requires additional data sources
        andas = []

        # For now, return empty - this requires more complex FDA database queries
        # In production, would use FDA's ANDA database or DrugPatentWatch
        logger.info(f"Searching for ANDAs referencing NDA {nda_number}")

        return andas

    def get_approved_generics_for_drug(
        self, brand_name: str = None, active_ingredient: str = None
    ) -> List[ANDAApplication]:
        """
        Get approved generic versions of a drug.

        Args:
            brand_name: Brand name of the reference drug.
            active_ingredient: Active ingredient to search for.

        Returns:
            List of approved ANDA applications.
        """
        if self._orange_book_products is None:
            self._orange_book_products = self._load_orange_book_anda()

        if self._orange_book_products.empty:
            return []

        # Filter by ingredient
        filtered = self._orange_book_products.copy()

        if active_ingredient:
            ingredient_upper = active_ingredient.upper()
            filtered = filtered[
                filtered["INGREDIENT"].str.upper().str.contains(
                    ingredient_upper, na=False
                )
            ]

        if brand_name:
            brand_upper = brand_name.upper()
            # For ANDAs, trade name often differs from brand
            filtered = filtered[
                filtered["INGREDIENT"].str.upper().str.contains(
                    brand_upper.split()[0][:4], na=False
                )
                | filtered["TRADE_NAME"].str.upper().str.contains(
                    brand_upper, na=False
                )
            ]

        andas = []
        for _, row in filtered.iterrows():
            approval_date = None
            if "APPROVAL_DATE" in row and row["APPROVAL_DATE"]:
                try:
                    approval_date = datetime.strptime(
                        str(row["APPROVAL_DATE"]), "%b %d, %Y"
                    ).date()
                except (ValueError, TypeError):
                    try:
                        approval_date = pd.to_datetime(row["APPROVAL_DATE"]).date()
                    except Exception:
                        pass

            anda = ANDAApplication(
                anda_number=str(row.get("APPL_NO", "")),
                drug_name=str(row.get("TRADE_NAME", "")),
                active_ingredient=str(row.get("INGREDIENT", "")),
                dosage_form=str(row.get("DF_ROUTE", "")).split(";")[0] if row.get("DF_ROUTE") else "",
                strength=str(row.get("STRENGTH", "")),
                applicant_company=str(row.get("APPLICANT", "")),
                approval_date=approval_date,
                tentative_approval_date=None,
                first_to_file=False,  # Not available in Orange Book
                paragraph_iv=False,  # Not available in Orange Book
                status="APPROVED" if approval_date else "UNKNOWN",
                reference_nda=None,
            )
            andas.append(anda)

        logger.info(
            f"Found {len(andas)} ANDA records for "
            f"ingredient={active_ingredient}, brand={brand_name}"
        )
        return andas

    def get_generic_company_ticker(self, company_name: str) -> Optional[str]:
        """
        Get the stock ticker for a generic pharmaceutical company.

        Args:
            company_name: Company name to look up.

        Returns:
            Stock ticker or None if not found/private.
        """
        company_upper = company_name.upper()

        for company_key, ticker in self.GENERIC_COMPANIES.items():
            if company_key in company_upper:
                return ticker

        return None

    def count_anda_filers(self, active_ingredient: str) -> int:
        """
        Count the number of companies that have filed ANDAs for an ingredient.

        Args:
            active_ingredient: Active ingredient name.

        Returns:
            Number of unique ANDA filers.
        """
        andas = self.get_approved_generics_for_drug(active_ingredient=active_ingredient)

        unique_companies = set()
        for anda in andas:
            # Normalize company name
            company = anda.applicant_company.upper().split()[0] if anda.applicant_company else ""
            if company:
                unique_companies.add(company)

        return len(unique_companies)

    def extract_for_database(
        self, nda_numbers: List[str] = None, active_ingredients: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract ANDA data in a format suitable for database loading.

        Args:
            nda_numbers: List of NDA numbers to find generics for.
            active_ingredients: List of active ingredients to search.

        Returns:
            List of ANDA records as dictionaries.
        """
        all_andas = []
        seen_anda_numbers = set()

        # Search by active ingredients
        if active_ingredients:
            for ingredient in active_ingredients:
                andas = self.get_approved_generics_for_drug(active_ingredient=ingredient)
                for anda in andas:
                    if anda.anda_number not in seen_anda_numbers:
                        seen_anda_numbers.add(anda.anda_number)
                        all_andas.append(
                            {
                                "anda_number": anda.anda_number,
                                "generic_drug_name": anda.drug_name,
                                "active_ingredient": anda.active_ingredient,
                                "dosage_form": anda.dosage_form,
                                "strength": anda.strength,
                                "generic_company": anda.applicant_company,
                                "generic_company_ticker": self.get_generic_company_ticker(
                                    anda.applicant_company
                                ),
                                "filing_date": None,  # Not available
                                "tentative_approval_date": anda.tentative_approval_date,
                                "final_approval_date": anda.approval_date,
                                "first_to_file": anda.first_to_file,
                                "paragraph_iv_certification": anda.paragraph_iv,
                                "status": anda.status,
                                "reference_nda": anda.reference_nda,
                                "data_source": "FDA",
                            }
                        )

        logger.info(f"Extracted {len(all_andas)} ANDA records for database")
        return all_andas

    def get_first_to_file_status(self, active_ingredient: str) -> Dict[str, Any]:
        """
        Determine first-to-file status for generics of a drug.

        The first generic to file a complete ANDA with a Paragraph IV certification
        gets 180 days of marketing exclusivity.

        Args:
            active_ingredient: Active ingredient name.

        Returns:
            Dictionary with first-to-file information.
        """
        andas = self.get_approved_generics_for_drug(active_ingredient=active_ingredient)

        if not andas:
            return {
                "has_first_to_file": False,
                "first_to_file_company": None,
                "total_filers": 0,
            }

        # Sort by approval date to find first
        approved_andas = [a for a in andas if a.approval_date]
        if approved_andas:
            approved_andas.sort(key=lambda x: x.approval_date)
            first = approved_andas[0]

            return {
                "has_first_to_file": True,
                "first_to_file_company": first.applicant_company,
                "first_to_file_anda": first.anda_number,
                "first_approval_date": first.approval_date,
                "total_filers": len(set(a.applicant_company for a in andas)),
            }

        return {
            "has_first_to_file": False,
            "first_to_file_company": None,
            "total_filers": len(set(a.applicant_company for a in andas)),
        }


class GenericCompetitionAnalyzer:
    """
    Analyzes generic competition landscape for branded drugs.

    Provides insights on:
    - Number of generic filers
    - Market share estimates post-generic entry
    - Revenue erosion projections
    """

    # Typical generic erosion rates by number of competitors
    EROSION_RATES = {
        1: 0.50,  # 1 generic: 50% revenue loss for brand
        2: 0.65,  # 2 generics: 65% loss
        3: 0.75,  # 3 generics: 75% loss
        4: 0.80,  # 4+ generics: 80%+ loss
    }

    @staticmethod
    def estimate_revenue_erosion(
        num_generic_competitors: int, annual_revenue: int
    ) -> Dict[str, Any]:
        """
        Estimate revenue erosion based on number of generic competitors.

        Args:
            num_generic_competitors: Number of approved generics.
            annual_revenue: Current annual revenue of branded drug.

        Returns:
            Dictionary with erosion estimates.
        """
        if num_generic_competitors == 0:
            erosion_rate = 0
        elif num_generic_competitors == 1:
            erosion_rate = GenericCompetitionAnalyzer.EROSION_RATES[1]
        elif num_generic_competitors == 2:
            erosion_rate = GenericCompetitionAnalyzer.EROSION_RATES[2]
        elif num_generic_competitors == 3:
            erosion_rate = GenericCompetitionAnalyzer.EROSION_RATES[3]
        else:
            erosion_rate = GenericCompetitionAnalyzer.EROSION_RATES[4]

        revenue_loss = int(annual_revenue * erosion_rate)
        remaining_revenue = annual_revenue - revenue_loss
        generic_market = revenue_loss  # Total generic market opportunity

        return {
            "num_generic_competitors": num_generic_competitors,
            "erosion_rate": erosion_rate,
            "revenue_loss": revenue_loss,
            "remaining_branded_revenue": remaining_revenue,
            "generic_market_opportunity": generic_market,
            "per_generic_opportunity": (
                generic_market // num_generic_competitors
                if num_generic_competitors > 0
                else 0
            ),
        }

    @staticmethod
    def classify_opportunity(market_opportunity: int) -> str:
        """
        Classify the market opportunity tier.

        Args:
            market_opportunity: Market opportunity in USD.

        Returns:
            Opportunity tier classification.
        """
        if market_opportunity >= 1_000_000_000:
            return "BLOCKBUSTER"
        elif market_opportunity >= 500_000_000:
            return "HIGH_VALUE"
        elif market_opportunity >= 100_000_000:
            return "MEDIUM_VALUE"
        else:
            return "SMALL"


if __name__ == "__main__":
    # Test the ANDA extractor
    extractor = ANDAExtractor()

    print("\n=== Testing ANDA Extractor ===")

    # Test getting generics for a known drug (adalimumab/Humira)
    print("\n--- Generics for adalimumab ---")
    generics = extractor.get_approved_generics_for_drug(active_ingredient="adalimumab")

    for anda in generics[:5]:
        print(f"  ANDA {anda.anda_number}: {anda.applicant_company}")
        print(f"    Approval: {anda.approval_date}")

    # Test competition analysis
    print("\n--- Competition Analysis ---")
    analyzer = GenericCompetitionAnalyzer()

    erosion = analyzer.estimate_revenue_erosion(
        num_generic_competitors=3,
        annual_revenue=20_000_000_000,  # $20B
    )

    print(f"  Competitors: {erosion['num_generic_competitors']}")
    print(f"  Erosion Rate: {erosion['erosion_rate']*100:.0f}%")
    print(f"  Revenue Loss: ${erosion['revenue_loss']:,}")
    print(f"  Generic Market: ${erosion['generic_market_opportunity']:,}")
    print(
        f"  Per Generic: ${erosion['per_generic_opportunity']:,}"
    )
