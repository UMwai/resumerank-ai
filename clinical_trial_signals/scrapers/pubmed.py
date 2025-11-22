"""
PubMed and medRxiv/bioRxiv Preprint Scraper for Clinical Trial Signal Detection System.

Monitors scientific publications and preprints for clinical trial related signals.
"""
import logging
import re
import time
import xml.etree.ElementTree as ET
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
from utils.validation import validate_preprint, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PreprintRecord:
    """Parsed preprint/publication record."""
    source: str = ""  # pubmed, medrxiv, biorxiv
    external_id: str = ""  # PMID or DOI
    title: str = ""
    abstract: str = ""
    authors: List[str] = field(default_factory=list)
    publication_date: Optional[date] = None
    journal: str = ""
    url: str = ""
    keywords: List[str] = field(default_factory=list)
    mesh_terms: List[str] = field(default_factory=list)
    doi: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


class PubMedScraper:
    """
    Scraper for PubMed scientific publications.

    Uses NCBI E-utilities API:
    https://www.ncbi.nlm.nih.gov/books/NBK25501/
    """

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    # Clinical trial related MeSH terms
    CLINICAL_MESH_TERMS = [
        "Clinical Trial", "Clinical Trials, Phase III as Topic",
        "Clinical Trials, Phase II as Topic", "Drug Therapy",
        "Randomized Controlled Trial", "Treatment Outcome",
    ]

    def __init__(self, api_key: str = None):
        """
        Initialize PubMed scraper.

        Args:
            api_key: Optional NCBI API key for higher rate limits
        """
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ClinicalTrialSignals/1.0 (research@example.com)",
        })
        self.api_key = api_key
        self.timeout = config.scraper.request_timeout
        self.retry_config = RetryConfig(
            max_retries=config.scraper.max_retries,
            base_delay=1.0,
            max_delay=30.0,
        )

    def _make_request(
        self,
        url: str,
        params: Dict = None
    ) -> Optional[requests.Response]:
        """
        Make a rate-limited and retry-enabled request.

        Args:
            url: URL to request
            params: Query parameters

        Returns:
            Response object or None on failure
        """
        # Apply rate limiting
        limiter = get_rate_limiter("pubmed")
        limiter.acquire(timeout=60)

        # Add API key if available
        if params is None:
            params = {}
        if self.api_key:
            params["api_key"] = self.api_key

        try:
            response = retry_request(
                method="GET",
                url=url,
                session=self.session,
                config=self.retry_config,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response

        except RetryExhausted as e:
            logger.error(f"PubMed request failed after retries: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"PubMed request error: {e}")
            return None

    def search_publications(
        self,
        query: str,
        days_back: int = 30,
        limit: int = 50
    ) -> List[str]:
        """
        Search PubMed and return PMIDs.

        Args:
            query: Search query string
            days_back: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of PMIDs
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(limit),
            "retmode": "json",
            "sort": "pub_date",
            "datetype": "pdat",
            "reldate": str(days_back),
        }

        logger.info(f"Searching PubMed: {query[:50]}...")

        response = self._make_request(self.ESEARCH_URL, params)

        if not response:
            return []

        try:
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            logger.info(f"Found {len(pmids)} publications")
            return pmids

        except ValueError as e:
            logger.error(f"Failed to parse PubMed search response: {e}")
            return []

    def fetch_publications(self, pmids: List[str]) -> List[PreprintRecord]:
        """
        Fetch publication details for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            List of PreprintRecord objects
        """
        if not pmids:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }

        response = self._make_request(self.EFETCH_URL, params)

        if not response:
            return []

        try:
            return self._parse_pubmed_xml(response.text)
        except Exception as e:
            logger.error(f"Failed to parse PubMed fetch response: {e}")
            return []

    def _parse_pubmed_xml(self, xml_text: str) -> List[PreprintRecord]:
        """Parse PubMed XML response into PreprintRecord objects."""
        records = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                record = self._parse_single_article(article)
                if record:
                    records.append(record)

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")

        return records

    def _parse_single_article(self, article: ET.Element) -> Optional[PreprintRecord]:
        """Parse a single PubmedArticle element."""
        try:
            medline = article.find(".//MedlineCitation")
            if medline is None:
                return None

            # PMID
            pmid_elem = medline.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            # Article info
            article_info = medline.find(".//Article")
            if article_info is None:
                return None

            # Title
            title_elem = article_info.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""

            # Abstract
            abstract_texts = []
            for abs_text in article_info.findall(".//AbstractText"):
                if abs_text.text:
                    abstract_texts.append(abs_text.text)
            abstract = " ".join(abstract_texts)

            # Authors
            authors = []
            for author in article_info.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None and last_name.text:
                    name = last_name.text
                    if first_name is not None and first_name.text:
                        name = f"{first_name.text} {name}"
                    authors.append(name)

            # Journal
            journal_elem = article_info.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # Publication date
            pub_date = None
            date_elem = article_info.find(".//PubDate")
            if date_elem is not None:
                year = date_elem.find("Year")
                month = date_elem.find("Month")
                day = date_elem.find("Day")
                if year is not None and year.text:
                    year_val = int(year.text)
                    month_val = 1
                    day_val = 1
                    if month is not None and month.text:
                        try:
                            month_val = int(month.text)
                        except ValueError:
                            # Month might be abbreviated name
                            month_map = {
                                "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                                "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                                "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                            }
                            month_val = month_map.get(month.text[:3], 1)
                    if day is not None and day.text:
                        try:
                            day_val = int(day.text)
                        except ValueError:
                            pass
                    try:
                        pub_date = date(year_val, month_val, day_val)
                    except ValueError:
                        pub_date = date(year_val, 1, 1)

            # MeSH terms
            mesh_terms = []
            for mesh in medline.findall(".//MeshHeading/DescriptorName"):
                if mesh.text:
                    mesh_terms.append(mesh.text)

            # Keywords
            keywords = []
            for kw in medline.findall(".//KeywordList/Keyword"):
                if kw.text:
                    keywords.append(kw.text)

            # DOI
            doi = ""
            for article_id in article.findall(".//ArticleIdList/ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text or ""
                    break

            return PreprintRecord(
                source="pubmed",
                external_id=pmid,
                title=title,
                abstract=abstract[:5000] if abstract else "",
                authors=authors[:20],  # Limit authors
                publication_date=pub_date,
                journal=journal,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                keywords=keywords,
                mesh_terms=mesh_terms,
                doi=doi,
                raw_data={},
            )

        except Exception as e:
            logger.warning(f"Failed to parse article: {e}")
            return None

    def search_clinical_trial_publications(
        self,
        drug_name: str = None,
        company_name: str = None,
        nct_id: str = None,
        indication: str = None,
        days_back: int = 30
    ) -> Tuple[List[PreprintRecord], List[Dict[str, Any]]]:
        """
        Search for clinical trial related publications.

        Args:
            drug_name: Drug or compound name
            company_name: Company/sponsor name
            nct_id: Clinical trial NCT ID
            indication: Disease/indication
            days_back: Days to look back

        Returns:
            Tuple of (publications, signals)
        """
        query_parts = []

        if nct_id:
            query_parts.append(f'"{nct_id}"[Title/Abstract]')
        if drug_name:
            query_parts.append(f'"{drug_name}"[Title/Abstract]')
        if company_name:
            query_parts.append(f'"{company_name}"[Affiliation]')
        if indication:
            query_parts.append(f'"{indication}"[Title/Abstract]')

        # Add clinical trial filter
        query_parts.append(
            '("clinical trial"[Publication Type] OR "phase 3"[Title/Abstract] OR '
            '"phase III"[Title/Abstract] OR "randomized controlled trial"[Publication Type])'
        )

        query = " AND ".join(query_parts) if query_parts else "clinical trial[Publication Type]"

        pmids = self.search_publications(query, days_back=days_back)
        publications = self.fetch_publications(pmids)

        # Detect signals
        signals = self._analyze_publications_for_signals(publications, drug_name, nct_id)

        return publications, signals

    def _analyze_publications_for_signals(
        self,
        publications: List[PreprintRecord],
        drug_name: str = None,
        nct_id: str = None
    ) -> List[Dict[str, Any]]:
        """Analyze publications for signals."""
        signals = []

        for pub in publications:
            text = f"{pub.title} {pub.abstract}".lower()

            # Positive signals
            positive_keywords = [
                ("met primary endpoint", "Primary endpoint met"),
                ("positive results", "Positive results reported"),
                ("statistically significant", "Statistical significance achieved"),
                ("superior to", "Superiority demonstrated"),
                ("breakthrough therapy", "Breakthrough therapy mentioned"),
            ]

            for keyword, description in positive_keywords:
                if keyword in text:
                    signals.append({
                        "signal_type": "preprint_positive",
                        "description": f"Publication: {description}",
                        "title": pub.title[:100],
                        "pmid": pub.external_id,
                        "publication_date": pub.publication_date.isoformat() if pub.publication_date else None,
                    })
                    break  # One signal per publication

            # Negative signals
            negative_keywords = [
                ("did not meet", "Endpoint not met"),
                ("failed to demonstrate", "Failed to demonstrate efficacy"),
                ("terminated", "Trial termination reported"),
                ("safety concern", "Safety concerns raised"),
                ("adverse events", "Adverse events reported"),
            ]

            for keyword, description in negative_keywords:
                if keyword in text:
                    signals.append({
                        "signal_type": "preprint_negative",
                        "description": f"Publication: {description}",
                        "title": pub.title[:100],
                        "pmid": pub.external_id,
                        "publication_date": pub.publication_date.isoformat() if pub.publication_date else None,
                    })
                    break

        return signals


class MedRxivScraper:
    """
    Scraper for medRxiv and bioRxiv preprints.

    Uses the medRxiv/bioRxiv API:
    https://api.medrxiv.org/
    """

    MEDRXIV_API = "https://api.medrxiv.org/details/medrxiv"
    BIORXIV_API = "https://api.biorxiv.org/details/biorxiv"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ClinicalTrialSignals/1.0 (research@example.com)",
        })
        self.timeout = config.scraper.request_timeout
        self.retry_config = RetryConfig(
            max_retries=config.scraper.max_retries,
            base_delay=1.0,
            max_delay=30.0,
        )

    def _make_request(self, url: str) -> Optional[Dict]:
        """Make a rate-limited request."""
        limiter = get_rate_limiter("medrxiv")
        limiter.acquire(timeout=60)

        try:
            response = retry_request(
                method="GET",
                url=url,
                session=self.session,
                config=self.retry_config,
                timeout=self.timeout,
            )
            response.raise_for_status()
            return response.json()

        except RetryExhausted as e:
            logger.error(f"medRxiv request failed after retries: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"medRxiv request error: {e}")
            return None

    def fetch_recent_preprints(
        self,
        server: str = "medrxiv",
        days_back: int = 7
    ) -> List[PreprintRecord]:
        """
        Fetch recent preprints from medRxiv or bioRxiv.

        Args:
            server: "medrxiv" or "biorxiv"
            days_back: Number of days to look back

        Returns:
            List of PreprintRecord objects
        """
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        base_url = self.MEDRXIV_API if server == "medrxiv" else self.BIORXIV_API
        url = f"{base_url}/{start_date.isoformat()}/{end_date.isoformat()}/0/json"

        logger.info(f"Fetching {server} preprints from {start_date} to {end_date}")

        data = self._make_request(url)

        if not data:
            return []

        preprints = []
        for item in data.get("collection", []):
            record = self._parse_preprint(item, server)
            if record:
                preprints.append(record)

        logger.info(f"Found {len(preprints)} preprints from {server}")
        return preprints

    def _parse_preprint(self, data: Dict, server: str) -> Optional[PreprintRecord]:
        """Parse preprint data from API response."""
        try:
            # Parse date
            pub_date = None
            date_str = data.get("date", "")
            if date_str:
                try:
                    pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except ValueError:
                    pass

            # Parse authors (semicolon separated string)
            authors_str = data.get("authors", "")
            authors = [a.strip() for a in authors_str.split(";") if a.strip()]

            return PreprintRecord(
                source=server,
                external_id=data.get("doi", ""),
                title=data.get("title", ""),
                abstract=data.get("abstract", "")[:5000],
                authors=authors[:20],
                publication_date=pub_date,
                journal=f"{server} preprint",
                url=f"https://www.{server}.org/content/{data.get('doi', '')}",
                doi=data.get("doi", ""),
                raw_data=data,
            )

        except Exception as e:
            logger.warning(f"Failed to parse preprint: {e}")
            return None

    def search_clinical_preprints(
        self,
        keywords: List[str] = None,
        days_back: int = 14
    ) -> Tuple[List[PreprintRecord], List[Dict[str, Any]]]:
        """
        Search for clinical trial related preprints.

        Args:
            keywords: Keywords to filter by
            days_back: Days to look back

        Returns:
            Tuple of (preprints, signals)
        """
        if keywords is None:
            keywords = [
                "clinical trial", "phase 3", "phase III", "randomized",
                "efficacy", "primary endpoint", "therapeutic"
            ]

        # Fetch from both servers
        medrxiv_preprints = self.fetch_recent_preprints("medrxiv", days_back)
        biorxiv_preprints = self.fetch_recent_preprints("biorxiv", days_back)

        all_preprints = medrxiv_preprints + biorxiv_preprints

        # Filter by keywords
        filtered = []
        for preprint in all_preprints:
            text = f"{preprint.title} {preprint.abstract}".lower()
            if any(kw.lower() in text for kw in keywords):
                filtered.append(preprint)

        # Detect signals
        signals = []
        for preprint in filtered:
            text = f"{preprint.title} {preprint.abstract}".lower()

            # Check for late-breaking abstract (high-profile preprint)
            if "late-breaking" in text or "phase 3 results" in text:
                signals.append({
                    "signal_type": "late_breaking_abstract",
                    "description": f"Late-breaking preprint: {preprint.title[:80]}",
                    "doi": preprint.doi,
                    "source": preprint.source,
                    "publication_date": preprint.publication_date.isoformat() if preprint.publication_date else None,
                })

            # Check for positive/negative signals
            if any(kw in text for kw in ["positive", "met primary endpoint", "superior"]):
                signals.append({
                    "signal_type": "preprint_positive",
                    "description": f"Positive preprint: {preprint.title[:80]}",
                    "doi": preprint.doi,
                    "source": preprint.source,
                })
            elif any(kw in text for kw in ["negative", "did not meet", "failed"]):
                signals.append({
                    "signal_type": "preprint_negative",
                    "description": f"Negative preprint: {preprint.title[:80]}",
                    "doi": preprint.doi,
                    "source": preprint.source,
                })

        logger.info(f"Found {len(filtered)} clinical preprints, {len(signals)} signals")
        return filtered, signals


class PreprintMonitor:
    """
    Combined monitor for PubMed and preprint servers.

    Coordinates searches across multiple sources.
    """

    def __init__(self, pubmed_api_key: str = None):
        self.pubmed = PubMedScraper(api_key=pubmed_api_key)
        self.medrxiv = MedRxivScraper()

    def monitor_trial(
        self,
        trial_id: str,
        drug_name: str = None,
        indication: str = None,
        days_back: int = 30
    ) -> Tuple[List[PreprintRecord], List[Dict[str, Any]]]:
        """
        Monitor publications for a specific clinical trial.

        Args:
            trial_id: NCT ID
            drug_name: Drug name
            indication: Disease/indication
            days_back: Days to look back

        Returns:
            Tuple of (all publications, all signals)
        """
        all_pubs = []
        all_signals = []

        # Search PubMed
        try:
            pubs, signals = self.pubmed.search_clinical_trial_publications(
                drug_name=drug_name,
                nct_id=trial_id,
                indication=indication,
                days_back=days_back
            )
            all_pubs.extend(pubs)
            all_signals.extend(signals)
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")

        # Search preprint servers
        try:
            keywords = [trial_id] if trial_id else []
            if drug_name:
                keywords.append(drug_name)
            if indication:
                keywords.append(indication)

            if keywords:
                pubs, signals = self.medrxiv.search_clinical_preprints(
                    keywords=keywords,
                    days_back=days_back
                )
                all_pubs.extend(pubs)
                all_signals.extend(signals)
        except Exception as e:
            logger.error(f"Preprint search failed: {e}")

        return all_pubs, all_signals

    def fetch_all_recent(
        self,
        days_back: int = 7
    ) -> Tuple[List[PreprintRecord], List[Dict[str, Any]]]:
        """
        Fetch all recent clinical trial related publications.

        Args:
            days_back: Days to look back

        Returns:
            Tuple of (all publications, all signals)
        """
        return self.medrxiv.search_clinical_preprints(days_back=days_back)


if __name__ == "__main__":
    # Test the scrapers
    logging.basicConfig(level=logging.INFO)

    print("Testing PubMed scraper...")
    print("-" * 40)

    pubmed = PubMedScraper()

    # Test search
    print("\nSearching for Moderna clinical trial publications...")
    pubs, signals = pubmed.search_clinical_trial_publications(
        company_name="Moderna",
        days_back=60
    )

    print(f"Found {len(pubs)} publications, {len(signals)} signals")

    for pub in pubs[:3]:
        print(f"\nPMID: {pub.external_id}")
        print(f"  Title: {pub.title[:60]}...")
        print(f"  Date: {pub.publication_date}")
        print(f"  Journal: {pub.journal}")

    print("\n\nTesting medRxiv scraper...")
    print("-" * 40)

    medrxiv = MedRxivScraper()

    preprints, signals = medrxiv.search_clinical_preprints(days_back=7)
    print(f"Found {len(preprints)} clinical preprints, {len(signals)} signals")

    for preprint in preprints[:3]:
        print(f"\nDOI: {preprint.doi}")
        print(f"  Title: {preprint.title[:60]}...")
        print(f"  Date: {preprint.publication_date}")
        print(f"  Source: {preprint.source}")
