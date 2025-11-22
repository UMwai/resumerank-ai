"""
Tests for AI Patent Claim Strength Analysis module.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ai.patent_analyzer import (
    PatentClaimAnalyzer,
    ClaimAnalysisResult,
    generate_sample_analyses,
)


class TestPatentClaimAnalyzer:
    """Tests for PatentClaimAnalyzer class."""

    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict('os.environ', {}, clear=True):
            analyzer = PatentClaimAnalyzer(api_key=None)
            assert analyzer.api_key is None
            assert analyzer._client is None

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        analyzer = PatentClaimAnalyzer(api_key="test-key")
        assert analyzer.api_key == "test-key"

    def test_analyze_patent_claims_mock(self):
        """Test mock analysis without API key."""
        analyzer = PatentClaimAnalyzer(api_key=None)

        result = analyzer.analyze_patent_claims(
            patent_number="6090382",
            drug_name="Humira",
            claims_text="1. A human antibody that binds to TNF-alpha comprising...",
        )

        assert isinstance(result, ClaimAnalysisResult)
        assert result.patent_number == "6090382"
        assert result.drug_name == "Humira"
        assert 0 <= result.overall_strength_score <= 100
        assert 0 <= result.upheld_probability <= 100
        assert result.claim_breadth in ["narrow", "moderate", "broad"]

    def test_analyze_with_metadata(self):
        """Test analysis with patent metadata."""
        analyzer = PatentClaimAnalyzer(api_key=None)

        metadata = {
            "filing_date": "2000-01-15",
            "issue_date": "2002-06-20",
            "expiration_date": "2023-01-31",
            "patent_type": "COMPOSITION",
            "therapeutic_area": "Immunology",
        }

        result = analyzer.analyze_patent_claims(
            patent_number="6090382",
            drug_name="Humira",
            claims_text="Claims text here...",
            patent_metadata=metadata,
        )

        assert result is not None
        assert result.patent_number == "6090382"

    def test_claim_breadth_detection(self):
        """Test claim breadth detection from text."""
        analyzer = PatentClaimAnalyzer(api_key=None)

        # Broad claims (using "comprising")
        broad_claims = """
        1. A pharmaceutical composition comprising an antibody.
        2. The composition of claim 1 including additional components.
        """
        result_broad = analyzer._create_mock_analysis(
            "TEST001", "TestDrug", broad_claims
        )
        assert result_broad.claim_breadth in ["broad", "moderate"]

        # Narrow claims (using "consisting of")
        narrow_claims = """
        1. A composition consisting of exactly compound X.
        2. The composition consisting essentially of compound X.
        """
        result_narrow = analyzer._create_mock_analysis(
            "TEST002", "TestDrug", narrow_claims
        )
        assert result_narrow.claim_breadth in ["narrow", "moderate"]

    def test_batch_analyze(self):
        """Test batch analysis of multiple patents."""
        analyzer = PatentClaimAnalyzer(api_key=None)

        patents = [
            {
                "patent_number": "6090382",
                "drug_name": "Humira",
                "claims_text": "Claims...",
            },
            {
                "patent_number": "7371746",
                "drug_name": "Eliquis",
                "claims_text": "Claims...",
            },
        ]

        results = analyzer.batch_analyze(patents)

        assert len(results) == 2
        assert results[0].patent_number == "6090382"
        assert results[1].patent_number == "7371746"

    def test_compare_to_benchmarks(self):
        """Test benchmark comparison."""
        analyzer = PatentClaimAnalyzer(api_key=None)

        result = ClaimAnalysisResult(
            patent_number="TEST001",
            drug_name="TestDrug",
            claim_breadth="moderate",
            claim_breadth_score=50.0,
            independent_claims_count=5,
            dependent_claims_count=15,
            overall_strength_score=60.0,
            litigation_probability=70.0,
            upheld_probability=55.0,
            vulnerabilities=["Test vulnerability"],
            vulnerability_severity="medium",
            prior_art_risk="medium",
            prior_art_concerns=["Test concern"],
            similar_patents=[],
            benchmark_percentile=50.0,
            key_claims_summary="Test summary",
            novelty_assessment="Test",
            obviousness_assessment="Test",
            enablement_assessment="Test",
            recommendations=["Test recommendation"],
        )

        comparison = analyzer.compare_to_benchmarks(result)

        assert "patent_number" in comparison
        assert "predicted_outcome" in comparison
        assert "most_similar_cases" in comparison
        assert len(comparison["most_similar_cases"]) <= 3

    def test_result_to_dict(self):
        """Test ClaimAnalysisResult serialization."""
        result = ClaimAnalysisResult(
            patent_number="TEST001",
            drug_name="TestDrug",
            claim_breadth="moderate",
            claim_breadth_score=50.0,
            independent_claims_count=5,
            dependent_claims_count=15,
            overall_strength_score=60.0,
            litigation_probability=70.0,
            upheld_probability=55.0,
            vulnerabilities=["Test"],
            vulnerability_severity="medium",
            prior_art_risk="medium",
            prior_art_concerns=["Test"],
            similar_patents=[],
            benchmark_percentile=50.0,
            key_claims_summary="Test",
            novelty_assessment="Test",
            obviousness_assessment="Test",
            enablement_assessment="Test",
            recommendations=["Test"],
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["patent_number"] == "TEST001"
        assert "analysis_date" in result_dict

    def test_generate_sample_analyses(self):
        """Test sample analyses generation."""
        results = generate_sample_analyses()

        assert len(results) >= 5
        for result in results:
            assert isinstance(result, ClaimAnalysisResult)
            assert result.overall_strength_score >= 0
            assert result.overall_strength_score <= 100


class TestLitigationBenchmarks:
    """Tests for litigation benchmark data."""

    def test_benchmark_data_exists(self):
        """Test that benchmark data is populated."""
        analyzer = PatentClaimAnalyzer()

        assert len(analyzer.LITIGATION_BENCHMARKS) > 0

    def test_benchmark_data_structure(self):
        """Test benchmark data structure."""
        analyzer = PatentClaimAnalyzer()

        for drug, data in analyzer.LITIGATION_BENCHMARKS.items():
            assert "upheld" in data
            assert "outcome" in data
            assert "strength" in data
            assert isinstance(data["strength"], int)
