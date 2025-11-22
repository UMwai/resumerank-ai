"""
Tests for ML-enhanced Scoring Model.
"""

import pytest
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.transformers.scoring import (
    ScoringWeights,
    DrugPatentData,
    CertaintyScoreCalculator,
    PatentCliffScorer,
    MLScoringModel,
    EnhancedPatentCliffScorer,
    HISTORICAL_TRAINING_DATA,
    THERAPEUTIC_ADJUSTMENTS,
    backtest_scoring_model,
)


class TestMLScoringModel:
    """Tests for MLScoringModel class."""

    def test_init(self):
        """Test initialization."""
        model = MLScoringModel()
        assert model is not None

    def test_training(self):
        """Test model training."""
        model = MLScoringModel()

        # Check training occurred
        metrics = model.get_metrics()
        assert metrics["training_samples"] > 0

    def test_predict(self):
        """Test prediction."""
        model = MLScoringModel()

        data = {
            "patent_ratio": 1.0,
            "days_to_expiration": 0,
            "active_litigation": 0,
            "resolved_litigation": 3,
            "patents_invalidated": 2,
            "approved_generics": 5,
            "pending_generics": 3,
            "pte_applied": 0,
            "pediatric_exclusivity": 0,
            "is_biologic": 0,
            "therapeutic_area_oncology": 0,
            "therapeutic_area_immunology": 1,
            "revenue_billions": 10.0,
        }

        score = model.predict(data)

        assert 0 <= score <= 100

    def test_optimized_weights(self):
        """Test optimized weights generation."""
        model = MLScoringModel()

        weights = model.get_optimized_weights()

        if weights is not None:
            # Weights should sum to approximately 1
            total = (
                weights.patent_expired +
                weights.no_litigation +
                weights.anda_approved +
                weights.no_extension
            )
            assert 0.95 <= total <= 1.05

    def test_feature_importances(self):
        """Test feature importance extraction."""
        model = MLScoringModel()
        metrics = model.get_metrics()

        if metrics.get("feature_importances"):
            importances = metrics["feature_importances"]
            assert len(importances) > 0
            # All importances should be non-negative
            for imp in importances.values():
                assert imp >= 0


class TestEnhancedPatentCliffScorer:
    """Tests for EnhancedPatentCliffScorer class."""

    def test_init_with_ml(self):
        """Test initialization with ML enabled."""
        scorer = EnhancedPatentCliffScorer(use_ml=True)
        assert scorer.ml_model is not None or not scorer.use_ml

    def test_init_without_ml(self):
        """Test initialization without ML."""
        scorer = EnhancedPatentCliffScorer(use_ml=False)
        assert scorer.ml_model is None

    def test_score_patent_cliff_enhanced(self):
        """Test enhanced scoring."""
        scorer = EnhancedPatentCliffScorer(use_ml=True)

        data = DrugPatentData(
            drug_id=1,
            brand_name="TestDrug",
            generic_name="testdrug",
            branded_company="TestCompany",
            branded_company_ticker="TEST",
            annual_revenue=5000000000,
            patent_numbers=["US1234567"],
            earliest_expiration=date.today() + timedelta(days=180),
            latest_expiration=date.today() + timedelta(days=365),
            all_patents_expired=False,
            expiring_patents_count=1,
            total_patents_count=2,
            active_litigation_count=1,
            resolved_litigation_count=0,
            patents_invalidated=0,
            approved_generics_count=2,
            pending_generics_count=3,
            first_to_file_exists=True,
            pte_applied=False,
            pediatric_exclusivity=False,
        )

        result = scorer.score_patent_cliff_enhanced(
            data,
            therapeutic_area="Oncology",
            is_biologic=False,
        )

        assert "scoring" in result
        assert "final_certainty_score" in result["scoring"]
        assert "model_type" in result["scoring"]

    def test_therapeutic_adjustment_applied(self):
        """Test that therapeutic adjustment is applied."""
        scorer = EnhancedPatentCliffScorer(use_ml=False)

        data = DrugPatentData(
            drug_id=1,
            brand_name="TestDrug",
            generic_name="testdrug",
            branded_company="TestCompany",
            branded_company_ticker="TEST",
            annual_revenue=5000000000,
            patent_numbers=["US1234567"],
            earliest_expiration=date.today() + timedelta(days=180),
            latest_expiration=date.today() + timedelta(days=365),
            all_patents_expired=False,
            expiring_patents_count=2,
            total_patents_count=2,
            active_litigation_count=0,
            resolved_litigation_count=0,
            patents_invalidated=0,
            approved_generics_count=3,
            pending_generics_count=2,
            first_to_file_exists=True,
            pte_applied=False,
            pediatric_exclusivity=False,
        )

        result_oncology = scorer.score_patent_cliff_enhanced(
            data, therapeutic_area="Oncology"
        )
        result_cardio = scorer.score_patent_cliff_enhanced(
            data, therapeutic_area="Cardiovascular"
        )

        # Oncology should have lower adjustment (slower erosion)
        assert result_oncology["scoring"]["therapeutic_adjustment"] == -5
        assert result_cardio["scoring"]["therapeutic_adjustment"] == 5

    def test_get_model_metrics(self):
        """Test model metrics retrieval."""
        scorer = EnhancedPatentCliffScorer(use_ml=True)

        metrics = scorer.get_model_metrics()

        assert isinstance(metrics, dict)


class TestBacktesting:
    """Tests for backtesting functionality."""

    def test_backtest_runs(self):
        """Test that backtest runs without errors."""
        scorer = EnhancedPatentCliffScorer(use_ml=True)

        results = backtest_scoring_model(scorer)

        assert "test_samples" in results
        assert "mean_absolute_error" in results
        assert results["test_samples"] > 0

    def test_backtest_predictions(self):
        """Test backtest prediction details."""
        scorer = EnhancedPatentCliffScorer(use_ml=True)

        results = backtest_scoring_model(scorer)

        assert "predictions" in results
        for pred in results["predictions"]:
            assert "drug" in pred
            assert "predicted" in pred
            assert "actual" in pred
            assert "error" in pred


class TestHistoricalTrainingData:
    """Tests for historical training data."""

    def test_data_exists(self):
        """Test that training data exists."""
        assert len(HISTORICAL_TRAINING_DATA) > 0

    def test_data_structure(self):
        """Test training data structure."""
        required_fields = [
            "drug",
            "patent_ratio",
            "days_to_expiration",
            "active_litigation",
            "approved_generics",
            "outcome_score",
        ]

        for record in HISTORICAL_TRAINING_DATA:
            for field in required_fields:
                assert field in record

    def test_outcome_scores_valid(self):
        """Test that outcome scores are valid."""
        for record in HISTORICAL_TRAINING_DATA:
            assert 0 <= record["outcome_score"] <= 100


class TestTherapeuticAdjustments:
    """Tests for therapeutic area adjustments."""

    def test_adjustments_exist(self):
        """Test that adjustments are defined."""
        assert len(THERAPEUTIC_ADJUSTMENTS) > 0

    def test_adjustment_values(self):
        """Test that adjustments are reasonable."""
        for area, adjustment in THERAPEUTIC_ADJUSTMENTS.items():
            assert -20 <= adjustment <= 20

    def test_expected_areas(self):
        """Test that expected areas are included."""
        expected = ["Oncology", "Immunology", "Cardiovascular"]
        for area in expected:
            assert area in THERAPEUTIC_ADJUSTMENTS
