"""
Tests for the patent cliff scoring model.
"""

import pytest
from datetime import date
from dateutil.relativedelta import relativedelta

from src.transformers.scoring import (
    CertaintyScoreCalculator,
    PatentCliffScorer,
    DrugPatentData,
    ScoringWeights,
)


class TestCertaintyScoreCalculator:
    """Tests for the CertaintyScoreCalculator class."""

    @pytest.fixture
    def calculator(self):
        return CertaintyScoreCalculator()

    def test_patent_expiration_all_expired(self, calculator):
        """Test score when all patents are expired."""
        score = calculator.calculate_patent_expiration_score(
            all_expired=True,
            expiring_count=2,
            total_count=2,
        )
        assert score == 100.0

    def test_patent_expiration_none_expiring(self, calculator):
        """Test score when no patents are expiring soon."""
        score = calculator.calculate_patent_expiration_score(
            all_expired=False,
            expiring_count=0,
            total_count=2,
            days_until_expiration=400,
        )
        assert score < 50

    def test_patent_expiration_imminent(self, calculator):
        """Test score when expiration is imminent."""
        score = calculator.calculate_patent_expiration_score(
            all_expired=False,
            expiring_count=2,
            total_count=2,
            days_until_expiration=30,
        )
        assert score >= 90

    def test_litigation_no_active(self, calculator):
        """Test litigation score with no active cases."""
        score = calculator.calculate_litigation_score(
            active_litigation=0,
            resolved_litigation=0,
            patents_invalidated=0,
        )
        assert score == 100.0

    def test_litigation_active_cases(self, calculator):
        """Test litigation score with active cases."""
        score = calculator.calculate_litigation_score(
            active_litigation=2,
            resolved_litigation=0,
            patents_invalidated=0,
        )
        assert score < 50

    def test_anda_multiple_approved(self, calculator):
        """Test ANDA score with multiple approvals."""
        score = calculator.calculate_anda_score(
            approved_count=3,
            pending_count=2,
            first_to_file=True,
        )
        assert score == 100.0

    def test_anda_none_approved(self, calculator):
        """Test ANDA score with no approvals."""
        score = calculator.calculate_anda_score(
            approved_count=0,
            pending_count=0,
            first_to_file=False,
        )
        assert score == 0.0

    def test_extension_no_risk(self, calculator):
        """Test extension score with no extension risk."""
        score = calculator.calculate_extension_score(
            pte_applied=False,
            pediatric_exclusivity=False,
        )
        assert score == 100.0

    def test_extension_pte_applied(self, calculator):
        """Test extension score when PTE is applied."""
        score = calculator.calculate_extension_score(
            pte_applied=True,
            pediatric_exclusivity=False,
        )
        assert score < 100


class TestPatentCliffScorer:
    """Tests for the PatentCliffScorer class."""

    @pytest.fixture
    def scorer(self):
        return PatentCliffScorer()

    @pytest.fixture
    def high_certainty_drug(self):
        """Drug with high certainty patent cliff."""
        return DrugPatentData(
            drug_id=1,
            brand_name="TestDrug",
            generic_name="testdrug",
            branded_company="TestCo",
            branded_company_ticker="TEST",
            annual_revenue=5_000_000_000,
            patent_numbers=["US1234567"],
            earliest_expiration=date.today() + relativedelta(months=3),
            latest_expiration=date.today() + relativedelta(months=3),
            all_patents_expired=False,
            expiring_patents_count=1,
            total_patents_count=1,
            active_litigation_count=0,
            resolved_litigation_count=2,
            patents_invalidated=1,
            approved_generics_count=4,
            pending_generics_count=2,
            first_to_file_exists=True,
            pte_applied=False,
            pediatric_exclusivity=False,
        )

    @pytest.fixture
    def low_certainty_drug(self):
        """Drug with low certainty patent cliff."""
        return DrugPatentData(
            drug_id=2,
            brand_name="UncertainDrug",
            generic_name="uncertaindrug",
            branded_company="UncertainCo",
            branded_company_ticker="UNCR",
            annual_revenue=3_000_000_000,
            patent_numbers=["US7654321"],
            earliest_expiration=date.today() + relativedelta(months=18),
            latest_expiration=date.today() + relativedelta(months=24),
            all_patents_expired=False,
            expiring_patents_count=1,
            total_patents_count=3,
            active_litigation_count=2,
            resolved_litigation_count=0,
            patents_invalidated=0,
            approved_generics_count=0,
            pending_generics_count=1,
            first_to_file_exists=False,
            pte_applied=True,
            pediatric_exclusivity=True,
        )

    def test_high_certainty_score(self, scorer, high_certainty_drug):
        """Test that high certainty drugs get high scores."""
        result = scorer.score_patent_cliff(high_certainty_drug)
        assert result["scoring"]["final_certainty_score"] >= 80

    def test_low_certainty_score(self, scorer, low_certainty_drug):
        """Test that low certainty drugs get low scores."""
        result = scorer.score_patent_cliff(low_certainty_drug)
        assert result["scoring"]["final_certainty_score"] < 60

    def test_market_opportunity_blockbuster(self, scorer):
        """Test blockbuster classification."""
        opp = scorer.calculate_market_opportunity(
            annual_revenue=10_000_000_000,
            num_generics=3,
        )
        assert opp["opportunity_tier"] == "BLOCKBUSTER"

    def test_market_opportunity_small(self, scorer):
        """Test small opportunity classification."""
        opp = scorer.calculate_market_opportunity(
            annual_revenue=50_000_000,
            num_generics=2,
        )
        assert opp["opportunity_tier"] == "SMALL"

    def test_trade_recommendation_high_conf(self, scorer, high_certainty_drug):
        """Test that high certainty generates execute recommendation."""
        result = scorer.score_patent_cliff(high_certainty_drug)
        rec = result["trade_recommendation"]
        assert rec["confidence"] == "HIGH"
        assert "TRADE" in rec["recommendation"] or "POSITION" in rec["recommendation"]

    def test_trade_recommendation_low_conf(self, scorer, low_certainty_drug):
        """Test that low certainty generates avoid recommendation."""
        result = scorer.score_patent_cliff(low_certainty_drug)
        rec = result["trade_recommendation"]
        assert rec["confidence"] in ["LOW", "MEDIUM"]


class TestScoringWeights:
    """Tests for custom scoring weights."""

    def test_custom_weights(self):
        """Test calculator with custom weights."""
        weights = ScoringWeights(
            patent_expired=0.50,
            no_litigation=0.20,
            anda_approved=0.20,
            no_extension=0.10,
        )
        calculator = CertaintyScoreCalculator(weights)

        result = calculator.calculate_certainty_score(
            DrugPatentData(
                drug_id=1,
                brand_name="Test",
                generic_name="test",
                branded_company="TestCo",
                branded_company_ticker="TST",
                annual_revenue=1_000_000_000,
                patent_numbers=["US1111111"],
                earliest_expiration=date.today() + relativedelta(days=30),
                latest_expiration=date.today() + relativedelta(days=30),
                all_patents_expired=True,
                expiring_patents_count=1,
                total_patents_count=1,
                active_litigation_count=0,
                resolved_litigation_count=0,
                patents_invalidated=0,
                approved_generics_count=3,
                pending_generics_count=0,
                first_to_file_exists=True,
                pte_applied=False,
                pediatric_exclusivity=False,
            )
        )

        # With all positive factors, should be near 100
        assert result["final_certainty_score"] >= 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
