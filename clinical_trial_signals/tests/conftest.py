"""
Pytest configuration and fixtures for Clinical Trial Signal Detection tests.
"""
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing config
os.environ["DRY_RUN"] = "true"
os.environ["DB_PASSWORD"] = "test_password"
os.environ["EMAIL_ENABLED"] = "false"


@pytest.fixture
def mock_db_connection():
    """Mock database connection for tests."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value = []
    mock_conn.execute_many.return_value = None

    with patch("database.connection.DatabaseConnection", return_value=mock_conn):
        yield mock_conn


@pytest.fixture
def sample_trial_data() -> Dict:
    """Sample trial data for testing."""
    return {
        "trial_id": "NCT12345678",
        "company_ticker": "MRNA",
        "drug_name": "Test Drug",
        "indication": "Cancer",
        "phase": "PHASE3",
        "enrollment_target": 500,
        "enrollment_current": 250,
        "start_date": date(2024, 1, 1),
        "expected_completion": date(2025, 6, 1),
        "primary_completion_date": date(2025, 3, 1),
        "primary_endpoint": "Overall Survival",
        "status": "RECRUITING",
        "sponsor": "Test Pharma Inc",
        "study_type": "INTERVENTIONAL",
    }


@pytest.fixture
def sample_signal_data() -> Dict:
    """Sample signal data for testing."""
    return {
        "trial_id": "NCT12345678",
        "signal_type": "enrollment_increase",
        "signal_value": "Enrollment increased by 15%",
        "signal_weight": 2,
        "source": "change_detection",
    }


@pytest.fixture
def sample_company_data() -> Dict:
    """Sample company data for testing."""
    return {
        "ticker": "MRNA",
        "company_name": "Moderna Inc",
        "market_cap": 15000000000,
        "sector": "Biotechnology",
        "cik": "1682852",
    }


@pytest.fixture
def sample_patent_data() -> Dict:
    """Sample patent data for testing."""
    return {
        "patent_number": "US12345678",
        "application_number": "17/123,456",
        "title": "Novel mRNA Vaccine Composition",
        "abstract": "A composition comprising modified mRNA for therapeutic use",
        "filing_date": date(2023, 6, 1),
        "grant_date": date(2024, 9, 15),
        "assignee": "Moderna Inc",
    }


@pytest.fixture
def sample_preprint_data() -> Dict:
    """Sample preprint data for testing."""
    return {
        "source": "pubmed",
        "external_id": "38123456",
        "title": "Phase 3 Trial Results of Novel Cancer Treatment",
        "abstract": "This randomized controlled trial demonstrates statistically significant efficacy",
        "publication_date": date(2024, 10, 1),
        "journal": "New England Journal of Medicine",
    }


@pytest.fixture
def mock_requests_session():
    """Mock requests session for API tests."""
    with patch("requests.Session") as mock_session:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {}
        mock_response.text = ""
        mock_response.raise_for_status.return_value = None

        mock_session.return_value.get.return_value = mock_response
        mock_session.return_value.request.return_value = mock_response

        yield mock_session


@pytest.fixture
def mock_clinicaltrials_response() -> Dict:
    """Mock ClinicalTrials.gov API response."""
    return {
        "studies": [
            {
                "protocolSection": {
                    "identificationModule": {
                        "nctId": "NCT12345678",
                        "briefTitle": "Test Phase 3 Trial",
                    },
                    "statusModule": {
                        "overallStatus": "RECRUITING",
                        "startDateStruct": {"date": "2024-01-01"},
                        "completionDateStruct": {"date": "2025-06-01"},
                        "primaryCompletionDateStruct": {"date": "2025-03-01"},
                    },
                    "designModule": {
                        "phases": ["PHASE3"],
                        "enrollmentInfo": {"count": 500},
                    },
                    "sponsorCollaboratorsModule": {
                        "leadSponsor": {"name": "Test Pharma Inc"},
                    },
                    "conditionsModule": {
                        "conditions": ["Cancer"],
                    },
                    "armsInterventionsModule": {
                        "interventions": [{"name": "Test Drug", "type": "DRUG"}],
                    },
                }
            }
        ]
    }


@pytest.fixture
def mock_sec_response() -> Dict:
    """Mock SEC EDGAR API response."""
    return {
        "name": "Test Pharma Inc",
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q"],
                "filingDate": ["2024-10-01", "2024-09-30"],
                "accessionNumber": ["0001234567-24-000001", "0001234567-24-000002"],
                "primaryDocument": ["doc1.htm", "doc2.htm"],
                "primaryDocDescription": ["Report", "Quarterly Report"],
            }
        }
    }


@pytest.fixture
def mock_pubmed_xml_response() -> str:
    """Mock PubMed XML response."""
    return """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>38123456</PMID>
                <Article>
                    <ArticleTitle>Phase 3 Trial Results</ArticleTitle>
                    <Abstract>
                        <AbstractText>Positive results from clinical trial.</AbstractText>
                    </Abstract>
                    <AuthorList>
                        <Author>
                            <LastName>Smith</LastName>
                            <ForeName>John</ForeName>
                        </Author>
                    </AuthorList>
                    <Journal>
                        <Title>Nature Medicine</Title>
                    </Journal>
                    <PubDate>
                        <Year>2024</Year>
                        <Month>10</Month>
                        <Day>01</Day>
                    </PubDate>
                </Article>
                <MeshHeadingList>
                    <MeshHeading>
                        <DescriptorName>Clinical Trial</DescriptorName>
                    </MeshHeading>
                </MeshHeadingList>
            </MedlineCitation>
            <ArticleIdList>
                <ArticleId IdType="doi">10.1038/test</ArticleId>
            </ArticleIdList>
        </PubmedArticle>
    </PubmedArticleSet>
    """
