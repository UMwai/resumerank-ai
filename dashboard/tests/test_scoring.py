"""Tests for the scoring module."""

import pytest
import pandas as pd
from utils.scoring import (
    ScoreWeights,
    ScoreNormalizer,
    RecommendationEngine,
    CombinedScorer,
    SignalSource,
    Recommendation,
    calculate_combined_scores,
    get_recommendation,
)


class TestScoreWeights:
    """Tests for ScoreWeights dataclass."""

    def test_default_weights(self):
        """Test default weights sum to 1.0."""
        weights = ScoreWeights()
        total = weights.clinical_trials + weights.patent_intelligence + weights.insider_hiring
        assert abs(total - 1.0) < 0.001

    def test_custom_weights_normalization(self):
        """Test that custom weights are normalized."""
        weights = ScoreWeights(
            clinical_trials=0.5,
            patent_intelligence=0.5,
            insider_hiring=0.5,
        )
        total = weights.clinical_trials + weights.patent_intelligence + weights.insider_hiring
        assert abs(total - 1.0) < 0.001

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = ScoreWeights()
        d = weights.to_dict()
        assert SignalSource.CLINICAL_TRIALS.value in d
        assert SignalSource.PATENT_INTELLIGENCE.value in d
        assert SignalSource.INSIDER_HIRING.value in d


class TestScoreNormalizer:
    """Tests for ScoreNormalizer class."""

    def test_clinical_trials_normalization(self):
        """Test clinical trials score normalization (0-10 scale)."""
        # Score of 5 should normalize to 0.5
        normalized = ScoreNormalizer.normalize(5.0, SignalSource.CLINICAL_TRIALS.value)
        assert abs(normalized - 0.5) < 0.001

        # Score of 10 should normalize to 1.0
        normalized = ScoreNormalizer.normalize(10.0, SignalSource.CLINICAL_TRIALS.value)
        assert abs(normalized - 1.0) < 0.001

    def test_patent_intelligence_normalization(self):
        """Test patent intelligence score normalization (0-100 scale)."""
        # Score of 50 should normalize to 0.5
        normalized = ScoreNormalizer.normalize(50.0, SignalSource.PATENT_INTELLIGENCE.value)
        assert abs(normalized - 0.5) < 0.001

    def test_insider_hiring_normalization(self):
        """Test insider/hiring score normalization (-10 to +10 scale)."""
        # Score of 0 should normalize to 0.5
        normalized = ScoreNormalizer.normalize(0.0, SignalSource.INSIDER_HIRING.value)
        assert abs(normalized - 0.5) < 0.001

        # Score of 10 should normalize to 1.0
        normalized = ScoreNormalizer.normalize(10.0, SignalSource.INSIDER_HIRING.value)
        assert abs(normalized - 1.0) < 0.001

    def test_denormalize(self):
        """Test denormalization returns original scale."""
        original = 7.5
        normalized = ScoreNormalizer.normalize(original, SignalSource.CLINICAL_TRIALS.value)
        denormalized = ScoreNormalizer.denormalize(normalized, SignalSource.CLINICAL_TRIALS.value)
        assert abs(denormalized - original) < 0.001


class TestRecommendationEngine:
    """Tests for RecommendationEngine class."""

    def test_strong_buy_recommendation(self):
        """Test STRONG BUY recommendation."""
        engine = RecommendationEngine()
        rec = engine.get_recommendation(0.80, 0.75)
        assert rec == Recommendation.STRONG_BUY

    def test_buy_recommendation(self):
        """Test BUY recommendation."""
        engine = RecommendationEngine()
        rec = engine.get_recommendation(0.65, 0.60)
        assert rec == Recommendation.BUY

    def test_hold_recommendation(self):
        """Test HOLD recommendation."""
        engine = RecommendationEngine()
        rec = engine.get_recommendation(0.50, 0.50)
        assert rec == Recommendation.HOLD

    def test_sell_recommendation(self):
        """Test SELL recommendation."""
        engine = RecommendationEngine()
        rec = engine.get_recommendation(0.35, 0.60)
        assert rec == Recommendation.SELL

    def test_strong_sell_recommendation(self):
        """Test STRONG SELL recommendation."""
        engine = RecommendationEngine()
        rec = engine.get_recommendation(0.15, 0.75)
        assert rec == Recommendation.STRONG_SELL

    def test_low_confidence_moderates_recommendation(self):
        """Test that low confidence results in moderate recommendations."""
        engine = RecommendationEngine()
        # High score but low confidence should not be STRONG BUY
        rec = engine.get_recommendation(0.80, 0.30)
        assert rec != Recommendation.STRONG_BUY


class TestCombinedScorer:
    """Tests for CombinedScorer class."""

    def test_calculate_combined_score_empty_scores(self):
        """Test combined score calculation with empty scores."""
        scorer = CombinedScorer()
        result = scorer.calculate_combined_score("TEST", "Test Company", {})
        assert result.recommendation == Recommendation.NO_DATA
        assert result.signal_count == 0

    def test_calculate_combined_score_single_source(self):
        """Test combined score with single source."""
        scorer = CombinedScorer()
        scores = {
            SignalSource.CLINICAL_TRIALS.value: {
                'score': 8.0,
                'confidence': 0.8,
                'count': 3,
            }
        }
        result = scorer.calculate_combined_score("TEST", "Test Company", scores)
        assert result.combined_score > 0
        assert result.signal_count == 3
        assert len(result.contributing_sources) == 1

    def test_calculate_combined_score_all_sources(self):
        """Test combined score with all sources."""
        scorer = CombinedScorer()
        scores = {
            SignalSource.CLINICAL_TRIALS.value: {
                'score': 8.0,
                'confidence': 0.8,
                'count': 3,
            },
            SignalSource.PATENT_INTELLIGENCE.value: {
                'score': 75.0,
                'confidence': 0.7,
                'count': 2,
            },
            SignalSource.INSIDER_HIRING.value: {
                'score': 5.0,
                'confidence': 0.9,
                'count': 5,
            },
        }
        result = scorer.calculate_combined_score("TEST", "Test Company", scores)
        assert result.combined_score > 0
        assert result.signal_count == 10
        assert len(result.contributing_sources) == 3

    def test_calculate_batch_scores(self):
        """Test batch score calculation."""
        scorer = CombinedScorer()
        data = [
            {
                'ticker': 'AAPL',
                'company_name': 'Apple Inc.',
                'scores': {
                    SignalSource.CLINICAL_TRIALS.value: {
                        'score': 7.0,
                        'confidence': 0.8,
                        'count': 2,
                    }
                }
            },
            {
                'ticker': 'GOOGL',
                'company_name': 'Alphabet Inc.',
                'scores': {
                    SignalSource.PATENT_INTELLIGENCE.value: {
                        'score': 80.0,
                        'confidence': 0.7,
                        'count': 3,
                    }
                }
            }
        ]
        df = scorer.calculate_batch_scores(data)
        assert len(df) == 2
        assert 'ticker' in df.columns
        assert 'combined_score' in df.columns

    def test_get_high_confidence_alerts(self):
        """Test high confidence alert extraction."""
        scorer = CombinedScorer()
        df = pd.DataFrame({
            'ticker': ['HIGH', 'LOW', 'MID'],
            'company_name': ['High Co', 'Low Co', 'Mid Co'],
            'combined_score': [0.85, 0.15, 0.50],
            'confidence': [0.80, 0.80, 0.80],
            'recommendation': ['STRONG BUY', 'STRONG SELL', 'HOLD'],
        })
        alerts = scorer.get_high_confidence_alerts(df)
        assert len(alerts) == 2  # One bullish, one bearish
        types = [a['type'] for a in alerts]
        assert 'BULLISH' in types
        assert 'BEARISH' in types


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_combined_scores_function(self):
        """Test convenience function for batch calculation."""
        data = [
            {
                'ticker': 'TEST',
                'company_name': 'Test Company',
                'scores': {
                    SignalSource.CLINICAL_TRIALS.value: {
                        'score': 7.0,
                        'confidence': 0.8,
                        'count': 1,
                    }
                }
            }
        ]
        df = calculate_combined_scores(data)
        assert len(df) == 1
        assert df.iloc[0]['ticker'] == 'TEST'

    def test_get_recommendation_function(self):
        """Test convenience function for single recommendation."""
        rec = get_recommendation(0.80, 0.75)
        assert rec == 'STRONG BUY'
