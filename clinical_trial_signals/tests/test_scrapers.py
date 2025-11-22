"""
Tests for scraper modules.
"""
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch, PropertyMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestClinicalTrialsScraper:
    """Tests for ClinicalTrialsScraper."""

    def test_parse_date_full(self):
        """Test parsing full date."""
        from scrapers.clinicaltrials import ClinicalTrialsScraper

        scraper = ClinicalTrialsScraper()
        result = scraper._parse_date({"date": "2024-06-15"})
        assert result == date(2024, 6, 15)

    def test_parse_date_month_only(self):
        """Test parsing date with month only."""
        from scrapers.clinicaltrials import ClinicalTrialsScraper

        scraper = ClinicalTrialsScraper()
        result = scraper._parse_date({"date": "2024-06"})
        assert result == date(2024, 6, 1)

    def test_parse_date_empty(self):
        """Test parsing empty date."""
        from scrapers.clinicaltrials import ClinicalTrialsScraper

        scraper = ClinicalTrialsScraper()
        result = scraper._parse_date({})
        assert result is None

    def test_parse_study_success(self, mock_clinicaltrials_response):
        """Test parsing study data successfully."""
        from scrapers.clinicaltrials import ClinicalTrialsScraper

        scraper = ClinicalTrialsScraper()
        study_data = mock_clinicaltrials_response["studies"][0]
        result = scraper._parse_study(study_data)

        assert result is not None
        assert result.nct_id == "NCT12345678"
        assert result.status == "RECRUITING"
        assert result.phase == "PHASE3"

    def test_search_phase3_trials_success(self, mock_clinicaltrials_response):
        """Test searching Phase 3 trials."""
        from scrapers.clinicaltrials import ClinicalTrialsScraper

        with patch.object(ClinicalTrialsScraper, "_make_request") as mock_request:
            mock_request.return_value = mock_clinicaltrials_response

            scraper = ClinicalTrialsScraper()
            results = scraper.search_phase3_trials(conditions=["cancer"], limit=10)

            assert len(results) == 1
            assert results[0].nct_id == "NCT12345678"


class TestSECEdgarScraper:
    """Tests for SECEdgarScraper."""

    def test_extract_8k_items(self):
        """Test extracting 8-K item numbers."""
        from scrapers.sec_edgar import SECEdgarScraper

        scraper = SECEdgarScraper()

        text = "Item 7.01 Regulation FD Disclosure Item 8.01 Other Events"
        items = scraper._extract_8k_items(text)

        assert "7.01" in items
        assert "8.01" in items

    def test_analyze_8k_positive_signal(self):
        """Test detecting positive signals in 8-K."""
        from scrapers.sec_edgar import SECEdgarScraper

        scraper = SECEdgarScraper()

        text = """
        The Company announced positive results from its Phase 3 clinical trial.
        The trial met its primary endpoint with statistical significance.
        """
        signals = scraper.analyze_8k_for_trial_signals(text, ["7.01"])

        assert len(signals) > 0
        assert any(s["signal_type"] == "sec_8k_positive" for s in signals)

    def test_analyze_8k_negative_signal(self):
        """Test detecting negative signals in 8-K."""
        from scrapers.sec_edgar import SECEdgarScraper

        scraper = SECEdgarScraper()

        text = """
        The Company's Phase 3 clinical trial failed to meet its primary endpoint.
        The trial did not achieve statistical significance.
        """
        signals = scraper.analyze_8k_for_trial_signals(text, ["8.01"])

        assert len(signals) > 0
        assert any(s["signal_type"] == "sec_8k_negative" for s in signals)

    def test_analyze_8k_non_trial_content(self):
        """Test that non-trial content doesn't generate signals."""
        from scrapers.sec_edgar import SECEdgarScraper

        scraper = SECEdgarScraper()

        text = """
        The Company announced a new office lease agreement.
        The lease is for 10 years starting January 2025.
        """
        signals = scraper.analyze_8k_for_trial_signals(text, ["8.01"])

        assert len(signals) == 0


class TestUSPTOScraper:
    """Tests for USPTOScraper."""

    def test_parse_patentsview_result(self):
        """Test parsing PatentsView API result."""
        from scrapers.uspto import USPTOScraper

        scraper = USPTOScraper()

        data = {
            "patent_number": "12345678",
            "patent_title": "Novel Drug Composition",
            "patent_abstract": "A pharmaceutical composition for treatment.",
            "patent_date": "2024-09-15",
            "patent_num_claims": 20,
            "inventors": [
                {"inventor_first_name": "John", "inventor_last_name": "Smith"},
            ],
            "assignees": [
                {"assignee_organization": "Test Pharma Inc"},
            ],
            "applications": [
                {"app_number": "17/123456", "app_date": "2023-06-01"},
            ],
        }

        result = scraper._parse_patentsview_result(data)

        assert result is not None
        assert result.patent_number == "US12345678"
        assert result.title == "Novel Drug Composition"
        assert result.grant_date == date(2024, 9, 15)
        assert len(result.inventors) == 1

    def test_analyze_patent_relevance(self):
        """Test analyzing patent relevance to clinical trials."""
        from scrapers.uspto import USPTOScraper, PatentRecord

        scraper = USPTOScraper()

        # High relevance patent
        patent = PatentRecord(
            title="Pharmaceutical Composition for Cancer Treatment",
            abstract="A therapeutic drug formulation for clinical trial use.",
        )
        relevance = scraper.analyze_patent_for_trial_relevance(patent)
        assert relevance > 0.5

        # Low relevance patent
        patent_low = PatentRecord(
            title="Office Chair Design",
            abstract="An ergonomic chair for office use.",
        )
        relevance_low = scraper.analyze_patent_for_trial_relevance(patent_low)
        assert relevance_low == 0.0


class TestPubMedScraper:
    """Tests for PubMedScraper."""

    def test_parse_pubmed_xml(self, mock_pubmed_xml_response):
        """Test parsing PubMed XML response."""
        from scrapers.pubmed import PubMedScraper

        scraper = PubMedScraper()
        records = scraper._parse_pubmed_xml(mock_pubmed_xml_response)

        assert len(records) == 1
        record = records[0]
        assert record.external_id == "38123456"
        assert record.source == "pubmed"
        assert record.title == "Phase 3 Trial Results"
        assert len(record.authors) == 1
        assert record.doi == "10.1038/test"

    def test_parse_single_article(self, mock_pubmed_xml_response):
        """Test parsing a single article element."""
        import xml.etree.ElementTree as ET
        from scrapers.pubmed import PubMedScraper

        scraper = PubMedScraper()
        root = ET.fromstring(mock_pubmed_xml_response)
        article = root.find(".//PubmedArticle")

        result = scraper._parse_single_article(article)

        assert result is not None
        assert result.journal == "Nature Medicine"


class TestMedRxivScraper:
    """Tests for MedRxivScraper."""

    def test_parse_preprint(self):
        """Test parsing medRxiv preprint data."""
        from scrapers.pubmed import MedRxivScraper

        scraper = MedRxivScraper()

        data = {
            "doi": "10.1101/2024.01.01.000001",
            "title": "Phase 3 Clinical Trial Results",
            "abstract": "Positive results from randomized trial.",
            "authors": "Smith, John; Doe, Jane",
            "date": "2024-10-01",
        }

        result = scraper._parse_preprint(data, "medrxiv")

        assert result is not None
        assert result.source == "medrxiv"
        assert result.doi == "10.1101/2024.01.01.000001"
        assert len(result.authors) == 2
        assert result.publication_date == date(2024, 10, 1)

    def test_search_clinical_preprints_signals(self):
        """Test signal detection from preprints."""
        from scrapers.pubmed import MedRxivScraper, PreprintRecord

        scraper = MedRxivScraper()

        # Mock the fetch method
        with patch.object(scraper, "fetch_recent_preprints") as mock_fetch:
            mock_fetch.return_value = [
                PreprintRecord(
                    source="medrxiv",
                    external_id="10.1101/test",
                    title="Phase 3 Results Show Positive Efficacy",
                    abstract="The trial met primary endpoint with statistical significance.",
                    publication_date=date(2024, 10, 1),
                    doi="10.1101/test",
                )
            ]

            preprints, signals = scraper.search_clinical_preprints(
                keywords=["Phase 3"],
                days_back=7
            )

            assert len(preprints) > 0
            # Should detect positive signal
            assert any(s.get("signal_type") == "preprint_positive" for s in signals)
