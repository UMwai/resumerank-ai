"""
FDA Orange Book Data Extractor

Extracts drug and patent information from the FDA Orange Book data files.
The Orange Book contains a list of FDA-approved drug products with therapeutic
equivalence evaluations, patent information, and exclusivity data.

Data source: https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
"""

import io
import os
import zipfile
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OrangeBookDrug:
    """Data class for Orange Book drug information."""

    ingredient: str
    df_route: str  # Dosage Form; Route of Administration
    trade_name: str
    applicant: str
    strength: str
    appl_type: str  # Application Type (N=NDA, A=ANDA)
    appl_no: str  # Application Number
    product_no: str
    te_code: str  # Therapeutic Equivalence Code
    approval_date: Optional[date]
    rld: str  # Reference Listed Drug
    rs: str  # Reference Standard
    type: str  # Type (RX=Prescription, OTC=Over-the-counter)
    applicant_full_name: Optional[str] = None


@dataclass
class OrangeBookPatent:
    """Data class for Orange Book patent information."""

    appl_type: str
    appl_no: str
    product_no: str
    patent_no: str
    patent_expire_date_text: str
    drug_substance_flag: str
    drug_product_flag: str
    patent_use_code: str
    delist_flag: str
    submission_date: Optional[date] = None

    @property
    def patent_expiration_date(self) -> Optional[date]:
        """Parse patent expiration date from text."""
        if not self.patent_expire_date_text:
            return None
        try:
            # Format is typically "Dec 31, 2025" or similar
            return datetime.strptime(
                self.patent_expire_date_text, "%b %d, %Y"
            ).date()
        except ValueError:
            try:
                # Try alternate format "12/31/2025"
                return datetime.strptime(
                    self.patent_expire_date_text, "%m/%d/%Y"
                ).date()
            except ValueError:
                logger.warning(
                    f"Could not parse patent expiration date: {self.patent_expire_date_text}"
                )
                return None


@dataclass
class OrangeBookExclusivity:
    """Data class for Orange Book exclusivity information."""

    appl_type: str
    appl_no: str
    product_no: str
    exclusivity_code: str
    exclusivity_date: Optional[date]


class OrangeBookExtractor:
    """
    Extracts and processes FDA Orange Book data.

    The Orange Book ZIP file contains several text files:
    - products.txt: Drug product information
    - patent.txt: Patent information
    - exclusivity.txt: Exclusivity information
    """

    ORANGE_BOOK_URL = "https://www.fda.gov/media/76860/download"
    BACKUP_URL = "https://www.accessdata.fda.gov/cder/ob.zip"

    # Top pharmaceutical companies by revenue (for filtering)
    TOP_PHARMA_COMPANIES = [
        "ABBVIE",
        "PFIZER",
        "JOHNSON",
        "MERCK",
        "BRISTOL",
        "LILLY",
        "NOVARTIS",
        "ROCHE",
        "ASTRAZENECA",
        "SANOFI",
        "TAKEDA",
        "AMGEN",
        "GILEAD",
        "BIOGEN",
        "REGENERON",
        "VERTEX",
        "MODERNA",
        "GSK",
        "GLAXO",
        "BOEHRINGER",
    ]

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl_hours: int = 24,
    ):
        """
        Initialize the Orange Book extractor.

        Args:
            cache_dir: Directory for caching downloaded files.
            use_cache: Whether to use cached data.
            cache_ttl_hours: Cache time-to-live in hours.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache/orange_book")
        self.use_cache = use_cache
        self.cache_ttl_hours = cache_ttl_hours

        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Data containers
        self._products_df: Optional[pd.DataFrame] = None
        self._patents_df: Optional[pd.DataFrame] = None
        self._exclusivity_df: Optional[pd.DataFrame] = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
    )
    def _download_orange_book(self) -> bytes:
        """
        Download the Orange Book ZIP file from FDA.

        Returns:
            ZIP file content as bytes.
        """
        logger.info("Downloading Orange Book data from FDA...")

        try:
            response = requests.get(
                self.ORANGE_BOOK_URL,
                timeout=120,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PatentIntelligence/1.0)"
                },
            )
            response.raise_for_status()
            logger.info(
                f"Orange Book downloaded successfully ({len(response.content) / 1024:.1f} KB)"
            )
            return response.content
        except requests.RequestException as e:
            logger.warning(f"Primary download failed: {e}, trying backup URL...")
            response = requests.get(
                self.BACKUP_URL,
                timeout=120,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PatentIntelligence/1.0)"
                },
            )
            response.raise_for_status()
            return response.content

    def _is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        cache_file = self.cache_dir / "ob.zip"

        if not cache_file.exists():
            return False

        # Check file age
        file_age_hours = (
            datetime.now().timestamp() - cache_file.stat().st_mtime
        ) / 3600

        return file_age_hours < self.cache_ttl_hours

    def _get_cached_data(self) -> Optional[bytes]:
        """Get cached Orange Book data if available and valid."""
        if not self.use_cache or not self._is_cache_valid():
            return None

        cache_file = self.cache_dir / "ob.zip"
        logger.info("Using cached Orange Book data")
        return cache_file.read_bytes()

    def _cache_data(self, data: bytes) -> None:
        """Cache downloaded Orange Book data."""
        cache_file = self.cache_dir / "ob.zip"
        cache_file.write_bytes(data)
        logger.info(f"Orange Book data cached to {cache_file}")

    def _extract_zip_contents(self, zip_data: bytes) -> Dict[str, pd.DataFrame]:
        """
        Extract and parse contents from Orange Book ZIP file.

        Args:
            zip_data: ZIP file content as bytes.

        Returns:
            Dictionary of DataFrames for each file type.
        """
        dataframes = {}

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            file_list = zf.namelist()
            logger.info(f"Orange Book ZIP contains: {file_list}")

            for filename in file_list:
                if filename.lower().endswith(".txt"):
                    logger.info(f"Processing {filename}...")
                    with zf.open(filename) as f:
                        # Read with various encoding fallbacks
                        try:
                            content = f.read().decode("utf-8")
                        except UnicodeDecodeError:
                            f.seek(0)
                            content = f.read().decode("latin-1")

                        # Parse as tilde-delimited file (Orange Book format)
                        df = pd.read_csv(
                            io.StringIO(content),
                            sep="~",
                            dtype=str,
                            on_bad_lines="skip",
                        )

                        # Clean column names
                        df.columns = df.columns.str.strip().str.upper()

                        base_name = Path(filename).stem.lower()
                        dataframes[base_name] = df
                        logger.info(f"Parsed {filename}: {len(df)} records")

        return dataframes

    def fetch_data(self) -> None:
        """
        Fetch and parse Orange Book data.

        Downloads from FDA or uses cache, then parses all data files.
        """
        # Try to use cached data
        zip_data = self._get_cached_data()

        if zip_data is None:
            zip_data = self._download_orange_book()
            self._cache_data(zip_data)

        # Extract and parse
        dataframes = self._extract_zip_contents(zip_data)

        # Store DataFrames
        self._products_df = dataframes.get("products")
        self._patents_df = dataframes.get("patent")
        self._exclusivity_df = dataframes.get("exclusivity")

        logger.info(
            f"Orange Book data loaded: "
            f"{len(self._products_df) if self._products_df is not None else 0} products, "
            f"{len(self._patents_df) if self._patents_df is not None else 0} patents, "
            f"{len(self._exclusivity_df) if self._exclusivity_df is not None else 0} exclusivity records"
        )

    def get_nda_products(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get NDA (New Drug Application) products only.

        These are branded/innovator drugs (not generics).

        Args:
            top_n: Limit to top N products (by applicant company size).

        Returns:
            DataFrame of NDA products.
        """
        if self._products_df is None:
            self.fetch_data()

        # Filter for NDAs only (exclude ANDAs which are generics)
        nda_df = self._products_df[self._products_df["APPL_TYPE"] == "N"].copy()

        # Filter for prescription drugs
        if "TYPE" in nda_df.columns:
            nda_df = nda_df[nda_df["TYPE"] == "RX"]

        # Parse approval date
        if "APPROVAL_DATE" in nda_df.columns:
            nda_df["APPROVAL_DATE_PARSED"] = pd.to_datetime(
                nda_df["APPROVAL_DATE"], errors="coerce"
            ).dt.date

        logger.info(f"Found {len(nda_df)} NDA products")

        if top_n:
            # Prioritize products from top pharma companies
            def company_priority(applicant):
                applicant_upper = str(applicant).upper()
                for i, company in enumerate(self.TOP_PHARMA_COMPANIES):
                    if company in applicant_upper:
                        return i
                return len(self.TOP_PHARMA_COMPANIES)

            nda_df["COMPANY_PRIORITY"] = nda_df["APPLICANT"].apply(company_priority)
            nda_df = nda_df.sort_values("COMPANY_PRIORITY").head(top_n)
            nda_df = nda_df.drop(columns=["COMPANY_PRIORITY"])

        return nda_df

    def get_patents_for_drug(self, appl_no: str) -> pd.DataFrame:
        """
        Get all patents associated with a drug application.

        Args:
            appl_no: NDA/ANDA application number.

        Returns:
            DataFrame of patents for the drug.
        """
        if self._patents_df is None:
            self.fetch_data()

        patents = self._patents_df[
            self._patents_df["APPL_NO"].astype(str) == str(appl_no)
        ].copy()

        # Parse patent expiration dates
        if "PATENT_EXPIRE_DATE_TEXT" in patents.columns:
            patents["PATENT_EXPIRATION_DATE"] = pd.to_datetime(
                patents["PATENT_EXPIRE_DATE_TEXT"], errors="coerce"
            ).dt.date

        return patents

    def get_exclusivity_for_drug(self, appl_no: str) -> pd.DataFrame:
        """
        Get exclusivity information for a drug application.

        Args:
            appl_no: NDA/ANDA application number.

        Returns:
            DataFrame of exclusivity records.
        """
        if self._exclusivity_df is None:
            self.fetch_data()

        exclusivity = self._exclusivity_df[
            self._exclusivity_df["APPL_NO"].astype(str) == str(appl_no)
        ].copy()

        # Parse exclusivity dates
        if "EXCLUSIVITY_DATE" in exclusivity.columns:
            exclusivity["EXCLUSIVITY_DATE_PARSED"] = pd.to_datetime(
                exclusivity["EXCLUSIVITY_DATE"], errors="coerce"
            ).dt.date

        return exclusivity

    def get_all_patents(self) -> pd.DataFrame:
        """
        Get all patent records from Orange Book.

        Returns:
            DataFrame of all patents.
        """
        if self._patents_df is None:
            self.fetch_data()

        patents = self._patents_df.copy()

        # Parse patent expiration dates
        if "PATENT_EXPIRE_DATE_TEXT" in patents.columns:
            patents["PATENT_EXPIRATION_DATE"] = pd.to_datetime(
                patents["PATENT_EXPIRE_DATE_TEXT"], errors="coerce"
            ).dt.date

        return patents

    def get_drugs_with_expiring_patents(
        self,
        months_ahead: int = 18,
        min_year: int = None,
        max_year: int = None,
    ) -> pd.DataFrame:
        """
        Get drugs with patents expiring within a specified timeframe.

        Args:
            months_ahead: Look ahead period in months.
            min_year: Minimum expiration year to include.
            max_year: Maximum expiration year to include.

        Returns:
            DataFrame of drugs with expiring patents.
        """
        if self._products_df is None or self._patents_df is None:
            self.fetch_data()

        # Get NDA products
        products = self.get_nda_products()

        # Get patents with expiration dates
        patents = self.get_all_patents()

        # Merge products with patents
        merged = pd.merge(
            products,
            patents,
            on=["APPL_TYPE", "APPL_NO", "PRODUCT_NO"],
            how="inner",
        )

        # Filter by expiration date
        today = date.today()

        if min_year is None:
            min_date = today
        else:
            min_date = date(min_year, 1, 1)

        if max_year is None:
            from dateutil.relativedelta import relativedelta

            max_date = today + relativedelta(months=months_ahead)
        else:
            max_date = date(max_year, 12, 31)

        # Filter for patents expiring in range
        merged = merged[
            (merged["PATENT_EXPIRATION_DATE"] >= min_date)
            & (merged["PATENT_EXPIRATION_DATE"] <= max_date)
        ]

        # Sort by expiration date
        merged = merged.sort_values("PATENT_EXPIRATION_DATE")

        logger.info(
            f"Found {len(merged)} drug-patent combinations expiring "
            f"between {min_date} and {max_date}"
        )

        return merged

    def extract_for_database(
        self, top_n: int = 50
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Extract data in a format suitable for database loading.

        Args:
            top_n: Number of top drugs to extract.

        Returns:
            Tuple of (drugs_list, patents_list, exclusivity_list).
        """
        if self._products_df is None:
            self.fetch_data()

        # Get top NDA products
        products = self.get_nda_products(top_n=top_n)

        drugs_list = []
        patents_list = []
        exclusivity_list = []

        seen_drugs = set()

        for _, row in products.iterrows():
            appl_no = str(row.get("APPL_NO", ""))

            # Avoid duplicates
            if appl_no in seen_drugs:
                continue
            seen_drugs.add(appl_no)

            # Extract drug info
            drug = {
                "nda_number": appl_no,
                "brand_name": str(row.get("TRADE_NAME", "")).strip(),
                "generic_name": str(row.get("INGREDIENT", "")).strip(),
                "active_ingredient": str(row.get("INGREDIENT", "")).strip(),
                "branded_company": str(row.get("APPLICANT_FULL_NAME", row.get("APPLICANT", ""))).strip(),
                "dosage_form": str(row.get("DF_ROUTE", "")).split(";")[0].strip() if row.get("DF_ROUTE") else None,
                "route_of_administration": str(row.get("DF_ROUTE", "")).split(";")[-1].strip() if row.get("DF_ROUTE") else None,
                "fda_approval_date": row.get("APPROVAL_DATE_PARSED"),
                "market_status": "ACTIVE",
            }
            drugs_list.append(drug)

            # Extract patents for this drug
            drug_patents = self.get_patents_for_drug(appl_no)
            for _, pat_row in drug_patents.iterrows():
                patent_no = str(pat_row.get("PATENT_NO", "")).strip()
                if not patent_no:
                    continue

                # Determine patent type from flags
                patent_type = "OTHER"
                if pat_row.get("DRUG_SUBSTANCE_FLAG") == "Y":
                    patent_type = "COMPOSITION"
                elif pat_row.get("DRUG_PRODUCT_FLAG") == "Y":
                    patent_type = "FORMULATION"
                elif pat_row.get("PATENT_USE_CODE"):
                    patent_type = "METHOD_OF_USE"

                patent = {
                    "nda_number": appl_no,
                    "patent_number": patent_no,
                    "patent_type": patent_type,
                    "patent_use_code": str(pat_row.get("PATENT_USE_CODE", "")).strip() or None,
                    "base_expiration_date": pat_row.get("PATENT_EXPIRATION_DATE"),
                    "patent_status": "ACTIVE"
                    if pat_row.get("DELIST_FLAG") != "Y"
                    else "DELISTED",
                    "data_source": "ORANGE_BOOK",
                }
                patents_list.append(patent)

            # Extract exclusivity for this drug
            drug_exclusivity = self.get_exclusivity_for_drug(appl_no)
            for _, excl_row in drug_exclusivity.iterrows():
                excl = {
                    "nda_number": appl_no,
                    "exclusivity_code": str(excl_row.get("EXCLUSIVITY_CODE", "")).strip(),
                    "exclusivity_date": excl_row.get("EXCLUSIVITY_DATE_PARSED"),
                }
                exclusivity_list.append(excl)

        logger.info(
            f"Extracted for database: {len(drugs_list)} drugs, "
            f"{len(patents_list)} patents, {len(exclusivity_list)} exclusivity records"
        )

        return drugs_list, patents_list, exclusivity_list

    def clear_cache(self) -> None:
        """Clear cached Orange Book data."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Orange Book cache cleared")


if __name__ == "__main__":
    # Test the extractor
    extractor = OrangeBookExtractor()
    extractor.fetch_data()

    print("\n=== Top 10 NDA Products ===")
    products = extractor.get_nda_products(top_n=10)
    print(products[["TRADE_NAME", "INGREDIENT", "APPLICANT"]].to_string())

    print("\n=== Drugs with Patents Expiring 2024-2026 ===")
    expiring = extractor.get_drugs_with_expiring_patents(min_year=2024, max_year=2026)
    print(
        expiring[
            ["TRADE_NAME", "INGREDIENT", "PATENT_NO", "PATENT_EXPIRATION_DATE"]
        ].head(20).to_string()
    )
