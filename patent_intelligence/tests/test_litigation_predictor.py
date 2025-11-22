"""
Tests for Litigation Outcome Prediction Model.
"""

import pytest
from datetime import date
import tempfile
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.litigation_predictor import (
    LitigationPredictor,
    LitigationPrediction,
    LitigationFeatures,
    HISTORICAL_LITIGATION_DATA,
)


class TestLitigationPredictor:
    """Tests for LitigationPredictor class."""

    def test_init_default(self):
        """Test default initialization."""
        predictor = LitigationPredictor()
        assert predictor is not None
        assert predictor.training_samples > 0

    def test_predict_basic(self):
        """Test basic prediction."""
        predictor = LitigationPredictor()

        features = LitigationFeatures(
            patent_age_at_litigation=10,
            claim_count=30,
            independent_claims=5,
            patent_type="COMPOSITION",
            court_jurisdiction="D. Del.",
            challenger_company="Teva",
            challenger_win_rate=0.55,
            annual_revenue=5000000000,
            therapeutic_area="Oncology",
            num_prior_challenges=2,
            prior_art_density=0.4,
            claim_breadth="moderate",
        )

        prediction = predictor.predict(
            patent_number="TEST001",
            drug_name="TestDrug",
            features=features,
        )

        assert isinstance(prediction, LitigationPrediction)
        assert prediction.patent_number == "TEST001"
        assert 0 <= prediction.upheld_probability <= 100
        assert 0 <= prediction.invalidated_probability <= 100
        assert prediction.predicted_outcome in ["UPHELD", "INVALIDATED"]

    def test_predict_high_strength(self):
        """Test prediction for strong patent."""
        predictor = LitigationPredictor()

        # Strong patent characteristics
        features = LitigationFeatures(
            patent_age_at_litigation=5,
            claim_count=50,
            independent_claims=10,
            patent_type="COMPOSITION",
            court_jurisdiction="D. Del.",
            challenger_company="Sandoz",
            challenger_win_rate=0.35,
            annual_revenue=8000000000,
            therapeutic_area="Oncology",
            num_prior_challenges=0,
            prior_art_density=0.2,
            claim_breadth="narrow",
            has_pte=True,
            is_biologic=True,
        )

        prediction = predictor.predict("STRONG001", "StrongDrug", features)

        # Strong patents should have higher upheld probability
        assert prediction.upheld_probability > 50

    def test_predict_weak_patent(self):
        """Test prediction for weak patent."""
        predictor = LitigationPredictor()

        # Weak patent characteristics
        features = LitigationFeatures(
            patent_age_at_litigation=18,
            claim_count=15,
            independent_claims=2,
            patent_type="METHOD_OF_USE",
            court_jurisdiction="E.D. Tex.",
            challenger_company="Teva",
            challenger_win_rate=0.55,
            annual_revenue=2000000000,
            therapeutic_area="Neurology",
            num_prior_challenges=5,
            prior_art_density=0.8,
            claim_breadth="broad",
        )

        prediction = predictor.predict("WEAK001", "WeakDrug", features)

        # Weak patents should have lower upheld probability
        assert prediction.invalidated_probability > 40

    def test_prediction_includes_risk_factors(self):
        """Test that predictions include risk factors."""
        predictor = LitigationPredictor()

        features = LitigationFeatures(
            patent_age_at_litigation=16,  # Old
            claim_count=20,
            independent_claims=3,
            patent_type="METHOD_OF_USE",  # Weaker type
            court_jurisdiction="D. Del.",
            challenger_company="Teva",
            challenger_win_rate=0.55,  # Strong challenger
            annual_revenue=3000000000,
            therapeutic_area="Neurology",
            num_prior_challenges=3,  # Multiple challenges
            prior_art_density=0.7,  # High prior art
            claim_breadth="broad",  # Broad claims
        )

        prediction = predictor.predict("TEST001", "TestDrug", features)

        assert len(prediction.risk_factors) > 0
        # Check for expected risk factors
        risk_text = " ".join(prediction.risk_factors).lower()
        assert "old" in risk_text or "prior art" in risk_text or "method" in risk_text

    def test_prediction_includes_similar_cases(self):
        """Test that predictions include similar cases."""
        predictor = LitigationPredictor()

        features = LitigationFeatures(
            patent_age_at_litigation=10,
            claim_count=30,
            independent_claims=5,
            patent_type="COMPOSITION",
            court_jurisdiction="D. Del.",
            challenger_company="Teva",
            challenger_win_rate=0.50,
            annual_revenue=5000000000,
            therapeutic_area="Oncology",
            num_prior_challenges=2,
            prior_art_density=0.4,
            claim_breadth="moderate",
        )

        prediction = predictor.predict("TEST001", "TestDrug", features)

        assert len(prediction.similar_cases) > 0
        for case in prediction.similar_cases:
            assert "drug" in case
            assert "outcome" in case

    def test_model_save_load(self):
        """Test model save and load."""
        predictor = LitigationPredictor()

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            predictor.save_model(temp_path)
            assert os.path.exists(temp_path)

            # Load model
            new_predictor = LitigationPredictor(model_path=temp_path)
            assert new_predictor.training_samples == predictor.training_samples
        finally:
            os.unlink(temp_path)

    def test_get_model_metrics(self):
        """Test model metrics retrieval."""
        predictor = LitigationPredictor()

        metrics = predictor.get_model_metrics()

        assert "model_type" in metrics
        assert "accuracy" in metrics
        assert "training_samples" in metrics
        assert "feature_count" in metrics
        assert metrics["training_samples"] > 0

    def test_prediction_to_dict(self):
        """Test prediction serialization."""
        predictor = LitigationPredictor()

        features = LitigationFeatures(
            patent_age_at_litigation=10,
            claim_count=30,
            independent_claims=5,
            patent_type="COMPOSITION",
            court_jurisdiction="D. Del.",
            challenger_company="Teva",
            challenger_win_rate=0.50,
            annual_revenue=5000000000,
            therapeutic_area="Oncology",
            num_prior_challenges=2,
            prior_art_density=0.4,
            claim_breadth="moderate",
        )

        prediction = predictor.predict("TEST001", "TestDrug", features)
        pred_dict = prediction.to_dict()

        assert isinstance(pred_dict, dict)
        assert "patent_number" in pred_dict
        assert "upheld_probability" in pred_dict
        assert "prediction_date" in pred_dict


class TestHistoricalData:
    """Tests for historical litigation data."""

    def test_historical_data_exists(self):
        """Test that historical data is populated."""
        assert len(HISTORICAL_LITIGATION_DATA) > 0

    def test_historical_data_structure(self):
        """Test historical data structure."""
        required_fields = [
            "patent_number",
            "drug_name",
            "outcome",
            "patent_age_at_litigation",
            "claim_count",
            "patent_type",
            "court_jurisdiction",
        ]

        for record in HISTORICAL_LITIGATION_DATA:
            for field in required_fields:
                assert field in record, f"Missing field: {field}"

    def test_outcome_values(self):
        """Test that outcomes are valid."""
        valid_outcomes = ["UPHELD", "INVALIDATED", "SETTLED"]

        for record in HISTORICAL_LITIGATION_DATA:
            assert record["outcome"] in valid_outcomes
