"""
Tests for the ML scoring module.
"""

import pytest
from datetime import date
from unittest.mock import Mock, patch
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ml_scorer import (
    MLSignalScorer,
    SignalFeatures,
    MLScore,
    MarketRegime,
    MarketRegimeDetector,
    SectorAdjuster,
    BiotechSector,
    generate_synthetic_training_data,
    SKLEARN_AVAILABLE,
)


class TestSignalFeatures:
    """Tests for SignalFeatures dataclass."""

    def test_default_creation(self):
        """Test default feature creation."""
        features = SignalFeatures()

        assert features.insider_buy_count == 0
        assert features.glassdoor_sentiment == 0.0
        assert features.market_regime == "neutral"

    def test_custom_creation(self):
        """Test custom feature creation."""
        features = SignalFeatures(
            insider_buy_count=3,
            ceo_transaction=1,
            commercial_jobs=5,
            glassdoor_sentiment=0.4
        )

        assert features.insider_buy_count == 3
        assert features.ceo_transaction == 1

    def test_to_array(self):
        """Test conversion to numpy array."""
        features = SignalFeatures(
            insider_buy_count=3,
            insider_sell_count=1,
            ceo_transaction=1
        )

        arr = features.to_array()

        assert isinstance(arr, np.ndarray)
        assert len(arr) == 31  # Total number of features

    def test_feature_names(self):
        """Test feature names list."""
        names = SignalFeatures.feature_names()

        assert isinstance(names, list)
        assert len(names) == 31
        assert 'insider_buy_count' in names
        assert 'ceo_transaction' in names


class TestMarketRegimeDetector:
    """Tests for MarketRegimeDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()

    def test_detect_bull_market(self):
        """Test bull market detection."""
        regime = self.detector.detect_regime(
            vix_level=12,
            spy_return_30d=5,
            xbi_return_30d=3
        )

        assert regime == MarketRegime.BULL

    def test_detect_bear_market(self):
        """Test bear market detection."""
        regime = self.detector.detect_regime(
            vix_level=28,
            spy_return_30d=-5,
            xbi_return_30d=-10
        )

        assert regime == MarketRegime.BEAR

    def test_detect_crisis(self):
        """Test crisis market detection."""
        regime = self.detector.detect_regime(
            vix_level=40,
            spy_return_30d=-15,
            xbi_return_30d=-25
        )

        assert regime == MarketRegime.CRISIS

    def test_detect_neutral(self):
        """Test neutral market detection."""
        regime = self.detector.detect_regime(
            vix_level=18,
            spy_return_30d=1,
            xbi_return_30d=0
        )

        assert regime == MarketRegime.NEUTRAL

    def test_regime_adjustments(self):
        """Test regime-specific adjustments."""
        bull_adj = self.detector.get_regime_adjustments(MarketRegime.BULL)
        bear_adj = self.detector.get_regime_adjustments(MarketRegime.BEAR)

        # Bull market should boost bullish signals
        assert bull_adj['bullish_multiplier'] > 1.0
        assert bull_adj['bearish_multiplier'] < 1.0

        # Bear market should boost bearish signals
        assert bear_adj['bearish_multiplier'] > 1.0
        assert bear_adj['bullish_multiplier'] < 1.0

    def test_caching(self):
        """Test regime caching."""
        # First call
        regime1 = self.detector.detect_regime(vix_level=15)

        # Second call should use cache
        regime2 = self.detector.detect_regime()  # No args

        assert regime1 == regime2


class TestSectorAdjuster:
    """Tests for SectorAdjuster."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adjuster = SectorAdjuster()

    def test_get_sector(self):
        """Test sector lookup."""
        assert self.adjuster.get_sector('CRSP') == BiotechSector.GENE_THERAPY
        assert self.adjuster.get_sector('ALNY') == BiotechSector.RARE_DISEASE
        assert self.adjuster.get_sector('NBIX') == BiotechSector.CNS
        assert self.adjuster.get_sector('UNKNOWN') == BiotechSector.OTHER

    def test_sector_weights(self):
        """Test sector-specific weights."""
        rare_disease = self.adjuster.get_sector_weights(BiotechSector.RARE_DISEASE)
        gene_therapy = self.adjuster.get_sector_weights(BiotechSector.GENE_THERAPY)

        # Rare disease should weight commercial jobs highly
        assert rare_disease['commercial_jobs_weight'] > 1.5

        # Gene therapy should weight insider buys highly
        assert gene_therapy['insider_buy_weight'] > 1.0


class TestMLSignalScorer:
    """Tests for MLSignalScorer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.scorer = MLSignalScorer()

    def test_initialization(self):
        """Test scorer initializes correctly."""
        assert self.scorer.regime_detector is not None
        assert self.scorer.sector_adjuster is not None
        assert self.scorer._is_trained == False

    def test_score_without_training(self):
        """Test scoring without training (uses rule-based)."""
        features = SignalFeatures(
            insider_buy_count=3,
            ceo_transaction=1,
            commercial_jobs=5,
            glassdoor_sentiment=0.4
        )

        score = self.scorer.score('MRNA', features)

        assert isinstance(score, MLScore)
        assert score.ticker == 'MRNA'
        assert -10 <= score.ml_score <= 10
        assert 0 <= score.confidence <= 1

    def test_score_with_market_data(self):
        """Test scoring with market context."""
        features = SignalFeatures(
            insider_buy_count=2,
            top_fund_new_positions=1
        )

        market_data = {'vix': 12, 'spy_return_30d': 5}

        score = self.scorer.score('MRNA', features, market_data)

        assert score.ml_score != 0  # Should have some signal

    def test_bullish_score(self):
        """Test bullish feature combination produces positive score."""
        features = SignalFeatures(
            insider_buy_count=5,
            ceo_transaction=1,
            multiple_insider_buy=True,
            top_fund_new_positions=3,
            institutional_convergence=True,
            commercial_jobs=8,
            glassdoor_sentiment=0.6
        )

        score = self.scorer.score('MRNA', features)

        assert score.ml_score > 0
        assert 'BUY' in score.recommendation.upper()

    def test_bearish_score(self):
        """Test bearish feature combination produces negative score."""
        features = SignalFeatures(
            insider_sell_count=5,
            ceo_transaction=-1,
            multiple_insider_sell=True,
            top_fund_exits=3,
            executive_departures=3,
            glassdoor_sentiment=-0.6,
            layoff_mentions=5
        )

        score = self.scorer.score('MRNA', features)

        assert score.ml_score < 0
        assert 'SELL' in score.recommendation.upper()

    def test_timeframe_scores(self):
        """Test multi-timeframe predictions."""
        features = SignalFeatures(
            insider_buy_count=3,
            ceo_transaction=1
        )

        score = self.scorer.score('MRNA', features)

        assert '1w' in score.timeframe_scores
        assert '1m' in score.timeframe_scores
        assert '3m' in score.timeframe_scores

    def test_recommendation_generation(self):
        """Test recommendation strings are generated."""
        features = SignalFeatures(insider_buy_count=1)

        score = self.scorer.score('MRNA', features)

        assert isinstance(score.recommendation, str)
        assert len(score.recommendation) > 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_training(self):
        """Test model training with synthetic data."""
        # Generate training data
        training_data = generate_synthetic_training_data(100)

        # Train
        metrics = self.scorer.train(training_data)

        assert self.scorer._is_trained == True
        assert '1m' in metrics

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn not installed")
    def test_scoring_after_training(self):
        """Test scoring after training."""
        training_data = generate_synthetic_training_data(100)
        self.scorer.train(training_data)

        features = SignalFeatures(
            insider_buy_count=3,
            ceo_transaction=1
        )

        score = self.scorer.score('MRNA', features)

        assert score.model_agreement > 0

    def test_insufficient_training_data(self):
        """Test handling of insufficient training data."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not installed")

        # Too few samples
        training_data = generate_synthetic_training_data(10)

        metrics = self.scorer.train(training_data)

        assert metrics['status'] == 'insufficient_data'

    def test_feature_importance(self):
        """Test feature importance extraction."""
        if not SKLEARN_AVAILABLE:
            pytest.skip("sklearn not installed")

        # Train with enough data
        training_data = generate_synthetic_training_data(200)
        self.scorer.train(training_data)

        importance = self.scorer.get_feature_importance()

        assert isinstance(importance, dict)


class TestSyntheticDataGeneration:
    """Tests for synthetic training data generation."""

    def test_generate_data(self):
        """Test synthetic data generation."""
        data = generate_synthetic_training_data(50)

        assert len(data) == 50
        assert all(isinstance(d[0], SignalFeatures) for d in data)
        assert all(isinstance(d[1], dict) for d in data)

    def test_data_structure(self):
        """Test generated data structure."""
        data = generate_synthetic_training_data(10)

        features, returns = data[0]

        assert hasattr(features, 'insider_buy_count')
        assert '1w' in returns
        assert '1m' in returns
        assert '3m' in returns


class TestMLScore:
    """Tests for MLScore dataclass."""

    def test_creation(self):
        """Test MLScore creation."""
        score = MLScore(
            ticker='MRNA',
            score_date=date.today(),
            ml_score=5.5,
            confidence=0.8,
            prediction_7d=2.5,
            prediction_30d=8.0,
            prediction_90d=12.0,
            feature_contributions={'insider_buys': 0.3},
            model_agreement=0.85,
            recommendation='Buy: Strong signals',
            timeframe_scores={'1w': 2.5, '1m': 5.0, '3m': 4.0}
        )

        assert score.ticker == 'MRNA'
        assert score.ml_score == 5.5
        assert score.confidence == 0.8


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
