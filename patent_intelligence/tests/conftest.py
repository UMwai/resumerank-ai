"""
Pytest configuration and shared fixtures for Patent Intelligence tests.
"""

import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, Any

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a temporary directory for test data."""
    return tmp_path_factory.mktemp("test_data")


@pytest.fixture
def sample_drug_record() -> Dict[str, Any]:
    """Provide a sample drug record for testing."""
    return {
        "drug_id": 1,
        "nda_number": "125057",
        "brand_name": "Humira",
        "generic_name": "adalimumab",
        "active_ingredient": "adalimumab",
        "branded_company": "AbbVie Inc.",
        "branded_company_ticker": "ABBV",
        "therapeutic_area": "Immunology",
        "dosage_form": "Injection",
        "route_of_administration": "Subcutaneous",
        "annual_revenue": 21237000000,
        "revenue_year": 2022,
        "fda_approval_date": date(2002, 12, 31),
        "market_status": "ACTIVE",
    }


@pytest.fixture
def sample_patent_record() -> Dict[str, Any]:
    """Provide a sample patent record for testing."""
    return {
        "patent_id": 1,
        "patent_number": "6090382",
        "drug_id": 1,
        "patent_type": "COMPOSITION",
        "patent_use_code": None,
        "filing_date": date(1996, 3, 15),
        "grant_date": date(2000, 7, 18),
        "base_expiration_date": date(2016, 3, 15),
        "pta_days": 0,
        "pte_days": 365,
        "adjusted_expiration_date": date(2017, 3, 15),
        "final_expiration_date": date(2023, 1, 31),
        "patent_status": "EXPIRED",
        "data_source": "ORANGE_BOOK",
    }


@pytest.fixture
def sample_litigation_record() -> Dict[str, Any]:
    """Provide a sample litigation record for testing."""
    return {
        "litigation_id": 1,
        "case_id": "1:17-cv-01065",
        "case_name": "AbbVie Inc. v. Sandoz Inc.",
        "patent_number": "6090382",
        "plaintiff": "AbbVie Inc.",
        "defendant": "Sandoz Inc.",
        "court": "District of Delaware",
        "jurisdiction": "District of Delaware",
        "case_type": "ANDA",
        "filing_date": date(2017, 8, 15),
        "outcome": "SETTLED",
        "data_source": "PACER",
    }


@pytest.fixture
def sample_anda_record() -> Dict[str, Any]:
    """Provide a sample ANDA record for testing."""
    return {
        "anda_id": 1,
        "anda_number": "207958",
        "drug_id": 1,
        "generic_company": "Amgen Inc.",
        "generic_company_ticker": "AMGN",
        "generic_drug_name": "adalimumab-atto",
        "dosage_form": "Injection",
        "filing_date": date(2017, 1, 15),
        "first_to_file": False,
        "paragraph_iv_certification": True,
        "final_approval_date": date(2023, 1, 31),
        "status": "APPROVED",
        "data_source": "FDA",
    }


@pytest.fixture
def sample_calendar_event() -> Dict[str, Any]:
    """Provide a sample calendar event for testing."""
    return {
        "event_id": 1,
        "drug_id": 1,
        "event_type": "PATENT_EXPIRATION",
        "event_date": date(2023, 1, 31),
        "related_patent_number": "6090382",
        "certainty_score": 95.0,
        "market_opportunity": 21237000000,
        "opportunity_tier": "BLOCKBUSTER",
        "trade_recommendation": "Consider short position on ABBV ahead of biosimilar launch",
        "recommendation_confidence": "HIGH",
    }


@pytest.fixture
def multiple_drug_records(sample_drug_record) -> list:
    """Provide multiple drug records for testing."""
    drugs = [sample_drug_record.copy()]

    # Add more drugs
    drugs.append({
        **sample_drug_record,
        "drug_id": 2,
        "nda_number": "125514",
        "brand_name": "Keytruda",
        "generic_name": "pembrolizumab",
        "active_ingredient": "pembrolizumab",
        "branded_company": "Merck & Co.",
        "branded_company_ticker": "MRK",
        "therapeutic_area": "Oncology",
        "annual_revenue": 25000000000,
        "revenue_year": 2023,
        "fda_approval_date": date(2014, 9, 4),
    })

    drugs.append({
        **sample_drug_record,
        "drug_id": 3,
        "nda_number": "202155",
        "brand_name": "Eliquis",
        "generic_name": "apixaban",
        "active_ingredient": "apixaban",
        "branded_company": "Bristol-Myers Squibb",
        "branded_company_ticker": "BMY",
        "therapeutic_area": "Cardiovascular",
        "annual_revenue": 12200000000,
        "revenue_year": 2023,
        "fda_approval_date": date(2012, 12, 28),
    })

    return drugs


@pytest.fixture
def mock_orange_book_data():
    """Provide mock Orange Book data for testing."""
    return {
        "products": [
            {
                "INGREDIENT": "ADALIMUMAB",
                "DF_ROUTE": "INJECTABLE; SUBCUTANEOUS",
                "TRADE_NAME": "HUMIRA",
                "APPLICANT": "ABBVIE",
                "STRENGTH": "10MG/0.1ML",
                "APPL_TYPE": "N",
                "APPL_NO": "125057",
                "PRODUCT_NO": "001",
                "TE_CODE": "BX",
                "APPROVAL_DATE": "Dec 31, 2002",
                "RLD": "Yes",
                "RS": "Yes",
                "TYPE": "RX",
            }
        ],
        "patents": [
            {
                "APPL_TYPE": "N",
                "APPL_NO": "125057",
                "PRODUCT_NO": "001",
                "PATENT_NO": "6090382",
                "PATENT_EXPIRE_DATE_TEXT": "Jan 31, 2023",
                "DRUG_SUBSTANCE_FLAG": "Y",
                "DRUG_PRODUCT_FLAG": "N",
                "PATENT_USE_CODE": "",
                "DELIST_FLAG": "N",
            }
        ],
    }


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch, tmp_path):
    """Set up test environment variables."""
    # Use temporary directories for caching
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))

    # Disable actual API calls in tests by default
    monkeypatch.setenv("PATENT_INTELLIGENCE_TEST_MODE", "true")


# Marker for slow tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_db: marks tests that require database connection"
    )
