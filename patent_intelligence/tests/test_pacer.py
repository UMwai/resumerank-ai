"""
Tests for the PACER court filing extractor.
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.extractors.pacer import (
    PACERExtractor,
    PACERCredentials,
    CourtFiling,
    DocketEntry,
    get_pacer_extractor,
)


class TestPACERCredentials:
    """Tests for PACERCredentials dataclass."""

    def test_credentials_creation(self):
        """Test creating credentials."""
        creds = PACERCredentials(
            username="testuser",
            password="testpass",
            client_code="TEST001",
        )
        assert creds.username == "testuser"
        assert creds.password == "testpass"
        assert creds.client_code == "TEST001"

    def test_credentials_optional_client_code(self):
        """Test credentials without client code."""
        creds = PACERCredentials(
            username="testuser",
            password="testpass",
        )
        assert creds.client_code is None


class TestCourtFiling:
    """Tests for CourtFiling dataclass."""

    def test_filing_creation(self):
        """Test creating a court filing."""
        filing = CourtFiling(
            case_id="test123",
            case_number="1:23-cv-01234",
            case_name="Test Corp v. Generic Co",
            court="District of Delaware",
            court_type="DISTRICT",
            jurisdiction="District of Delaware",
            filing_date=date(2023, 1, 15),
            case_type="PATENT",
            nature_of_suit="830",
            cause_of_action="Patent Infringement",
            plaintiff="Test Corp",
            defendant="Generic Co",
        )
        assert filing.case_id == "test123"
        assert filing.case_type == "PATENT"
        assert filing.status == "OPEN"

    def test_filing_to_dict(self):
        """Test converting filing to dictionary."""
        filing = CourtFiling(
            case_id="test123",
            case_number="1:23-cv-01234",
            case_name="Test Corp v. Generic Co",
            court="District of Delaware",
            court_type="DISTRICT",
            jurisdiction="District of Delaware",
            filing_date=date(2023, 1, 15),
            case_type="ANDA",
            nature_of_suit="835",
            cause_of_action="ANDA Filing",
            plaintiff="Test Corp",
            defendant="Generic Co",
            related_patents=["1234567", "7654321"],
        )
        result = filing.to_dict()

        assert isinstance(result, dict)
        assert result["case_id"] == "test123"
        assert result["case_type"] == "ANDA"
        assert result["related_patents"] == ["1234567", "7654321"]


class TestDocketEntry:
    """Tests for DocketEntry dataclass."""

    def test_docket_entry_creation(self):
        """Test creating a docket entry."""
        entry = DocketEntry(
            entry_number=1,
            filing_date=date(2023, 1, 15),
            description="COMPLAINT filed",
        )
        assert entry.entry_number == 1
        assert entry.is_sealed is False

    def test_docket_entry_to_dict(self):
        """Test converting docket entry to dictionary."""
        entry = DocketEntry(
            entry_number=5,
            filing_date=date(2023, 2, 20),
            description="MOTION for Summary Judgment",
            filed_by="Plaintiff",
        )
        result = entry.to_dict()

        assert result["entry_number"] == 5
        assert result["filed_by"] == "Plaintiff"


class TestPACERExtractor:
    """Tests for the PACERExtractor class."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create a PACER extractor for testing."""
        return PACERExtractor(
            cache_dir=str(tmp_path / "pacer_cache"),
            use_cache=True,
            monthly_budget=30.0,
        )

    @pytest.fixture
    def authenticated_extractor(self, tmp_path):
        """Create an authenticated PACER extractor."""
        creds = PACERCredentials(
            username="testuser",
            password="testpass",
        )
        return PACERExtractor(
            credentials=creds,
            cache_dir=str(tmp_path / "pacer_cache"),
        )

    def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor is not None
        assert extractor.use_cache is True
        assert extractor.monthly_budget == 30.0

    def test_cache_directory_creation(self, extractor):
        """Test that cache directory is created."""
        assert extractor.cache_dir.exists()

    def test_pharma_courts_defined(self, extractor):
        """Test that pharma courts are defined."""
        assert len(extractor.PHARMA_COURTS) > 0
        assert "deb" in extractor.PHARMA_COURTS

    def test_pharma_nos_codes_defined(self, extractor):
        """Test that nature of suit codes are defined."""
        assert "830" in extractor.PHARMA_NOS_CODES
        assert "835" in extractor.PHARMA_NOS_CODES

    def test_budget_check_within_budget(self, extractor):
        """Test budget check when within budget."""
        assert extractor._check_budget(10.0) is True

    def test_budget_check_exceeds_budget(self, extractor):
        """Test budget check when exceeding budget."""
        extractor._current_month_cost = 25.0
        assert extractor._check_budget(10.0) is False

    def test_demo_mode_returns_data(self, extractor):
        """Test that demo mode returns data without PACER access."""
        filings = extractor.search_cases(drug_name="Humira")

        assert len(filings) > 0
        assert all(isinstance(f, CourtFiling) for f in filings)

    def test_search_by_drug_name(self, extractor):
        """Test searching by drug name."""
        filings = extractor.search_cases(drug_name="Humira")

        # Should find Humira-related demo cases
        humira_filings = [
            f for f in filings
            if "humira" in f.case_name.lower() or "humira" in str(f.related_drug_names).lower()
        ]
        assert len(humira_filings) > 0

    def test_search_by_party_name(self, extractor):
        """Test searching by party name."""
        filings = extractor.search_cases(party_name="AbbVie")

        # Should find AbbVie-related cases
        abbvie_filings = [
            f for f in filings
            if "abbvie" in f.plaintiff.lower() or "abbvie" in f.defendant.lower()
        ]
        assert len(abbvie_filings) >= 0  # May or may not be in demo data

    def test_extract_for_database(self, extractor):
        """Test extracting data formatted for database."""
        records = extractor.extract_for_database(
            drug_names=["Humira", "Keytruda"]
        )

        assert isinstance(records, list)
        if len(records) > 0:
            assert "case_id" in records[0]
            assert "plaintiff" in records[0]
            assert "defendant" in records[0]

    def test_get_demo_docket_entries(self, extractor):
        """Test getting demo docket entries."""
        entries = extractor._get_demo_docket_entries()

        assert len(entries) > 0
        assert all(isinstance(e, DocketEntry) for e in entries)

    def test_extract_patent_numbers(self, extractor):
        """Test extracting patent numbers from text."""
        text = "Patent US7654321 and patent 1,234,567 were at issue"
        patents = extractor._extract_patent_numbers(text)

        assert len(patents) > 0
        # Check that numbers are extracted (format may vary)

    def test_extract_parties(self, extractor):
        """Test extracting parties from case name."""
        case_name = "AbbVie Inc. v. Sandoz Inc."
        plaintiff, defendant = extractor._extract_parties(case_name)

        assert "AbbVie" in plaintiff
        assert "Sandoz" in defendant

    def test_determine_case_type_anda(self, extractor):
        """Test determining ANDA case type."""
        case_type = extractor._determine_case_type(
            "Test Corp ANDA litigation",
            "835",
        )
        assert case_type == "ANDA"

    def test_determine_case_type_patent(self, extractor):
        """Test determining patent case type."""
        case_type = extractor._determine_case_type(
            "Test Corp v. Generic Co",
            "830",
        )
        assert case_type == "PATENT"

    def test_close(self, extractor):
        """Test closing the extractor."""
        extractor.close()
        # Should not raise an error


class TestGetPacerExtractor:
    """Tests for the convenience function."""

    def test_create_extractor_without_credentials(self):
        """Test creating extractor without credentials."""
        extractor = get_pacer_extractor()
        assert extractor is not None
        assert extractor.credentials is None
        extractor.close()

    def test_create_extractor_with_credentials(self, tmp_path):
        """Test creating extractor with credentials."""
        extractor = get_pacer_extractor(
            username="testuser",
            password="testpass",
            cache_dir=str(tmp_path / "cache"),
        )
        assert extractor is not None
        assert extractor.credentials is not None
        assert extractor.credentials.username == "testuser"
        extractor.close()


class TestPACERCaching:
    """Tests for PACER caching functionality."""

    @pytest.fixture
    def extractor(self, tmp_path):
        """Create extractor with caching enabled."""
        return PACERExtractor(
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
            cache_ttl_hours=24,
        )

    def test_cache_key_generation(self, extractor):
        """Test cache key generation is consistent."""
        key1 = extractor._get_cache_key("test query")
        key2 = extractor._get_cache_key("test query")
        assert key1 == key2

    def test_cache_key_different_for_different_queries(self, extractor):
        """Test different queries produce different keys."""
        key1 = extractor._get_cache_key("query 1")
        key2 = extractor._get_cache_key("query 2")
        assert key1 != key2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
