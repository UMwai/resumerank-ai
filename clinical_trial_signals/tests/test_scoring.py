"""
Tests for the signal scoring module.
"""
import pytest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scoring.signal_scorer import SignalScorer, Recommendation, ScoringResult


class TestSignalScorer:
    """Tests for SignalScorer class."""

    @pytest.fixture
    def scorer(self):
        """Create a SignalScorer instance."""
        return SignalScorer()

    def test_initialization(self, scorer):
        """Test SignalScorer initialization."""
        assert scorer.config is not None
        assert scorer.weights is not None
        assert scorer._max_positive_weight > 0
        assert scorer._max_negative_weight > 0

    def test_get_recommendation_strong_buy(self, scorer):
        """Test STRONG_BUY recommendation."""
        rec = scorer._get_recommendation(score=8.0, confidence=0.8)
        assert rec == Recommendation.STRONG_BUY

    def test_get_recommendation_buy(self, scorer):
        """Test BUY recommendation."""
        rec = scorer._get_recommendation(score=6.0, confidence=0.5)
        assert rec == Recommendation.BUY

    def test_get_recommendation_hold(self, scorer):
        """Test HOLD recommendation."""
        rec = scorer._get_recommendation(score=4.5, confidence=0.5)
        assert rec == Recommendation.HOLD

    def test_get_recommendation_short(self, scorer):
        """Test SHORT recommendation."""
        rec = scorer._get_recommendation(score=2.5, confidence=0.8)
        assert rec == Recommendation.SHORT

    def test_get_recommendation_strong_short(self, scorer):
        """Test STRONG_SHORT recommendation."""
        rec = scorer._get_recommendation(score=1.5, confidence=0.8)
        assert rec == Recommendation.STRONG_SHORT

    def test_calculate_confidence_no_signals(self, scorer):
        """Test confidence calculation with no signals."""
        confidence = scorer._calculate_confidence([])
        assert confidence == 0.0

    def test_calculate_confidence_single_signal(self, scorer):
        """Test confidence calculation with single signal."""
        mock_signal = MagicMock()
        mock_signal.signal_type = "test_signal"
        mock_signal.signal_weight = 3

        confidence = scorer._calculate_confidence([mock_signal])
        assert 0.0 < confidence <= 1.0

    def test_calculate_confidence_multiple_signals(self, scorer):
        """Test confidence increases with more signals."""
        mock_signal1 = MagicMock()
        mock_signal1.signal_type = "signal_type_1"
        mock_signal1.signal_weight = 3

        mock_signal2 = MagicMock()
        mock_signal2.signal_type = "signal_type_2"
        mock_signal2.signal_weight = 2

        confidence_1 = scorer._calculate_confidence([mock_signal1])
        confidence_2 = scorer._calculate_confidence([mock_signal1, mock_signal2])

        assert confidence_2 > confidence_1

    def test_calculate_confidence_consistency(self, scorer):
        """Test confidence is higher when signals are consistent."""
        # All positive signals
        positive_signals = [
            MagicMock(signal_type=f"pos_{i}", signal_weight=2)
            for i in range(3)
        ]

        # Mixed signals
        mixed_signals = [
            MagicMock(signal_type="pos_1", signal_weight=2),
            MagicMock(signal_type="neg_1", signal_weight=-2),
            MagicMock(signal_type="pos_2", signal_weight=2),
        ]

        conf_consistent = scorer._calculate_confidence(positive_signals)
        conf_mixed = scorer._calculate_confidence(mixed_signals)

        assert conf_consistent > conf_mixed

    def test_score_trial_no_signals(self, scorer):
        """Test scoring trial with no signals returns None."""
        with patch("scoring.signal_scorer.TrialSignal") as mock_signal:
            mock_signal.get_by_trial.return_value = []

            result = scorer.score_trial("NCT12345678")
            assert result is None

    def test_score_trial_with_signals(self, scorer):
        """Test scoring trial with signals."""
        mock_signal = MagicMock()
        mock_signal.signal_type = "enrollment_increase"
        mock_signal.signal_weight = 2
        mock_signal.signal_value = "Test signal"
        mock_signal.detected_date = datetime.now()

        with patch("scoring.signal_scorer.TrialSignal") as mock_signal_class:
            mock_signal_class.get_by_trial.return_value = [mock_signal]

            result = scorer.score_trial("NCT12345678")

            assert result is not None
            assert isinstance(result, ScoringResult)
            assert result.trial_id == "NCT12345678"
            assert result.signal_count == 1


class TestRecommendation:
    """Tests for Recommendation enum."""

    def test_recommendation_values(self):
        """Test recommendation enum values."""
        assert Recommendation.STRONG_BUY.value == "STRONG_BUY"
        assert Recommendation.BUY.value == "BUY"
        assert Recommendation.HOLD.value == "HOLD"
        assert Recommendation.SHORT.value == "SHORT"
        assert Recommendation.STRONG_SHORT.value == "STRONG_SHORT"


class TestScoringResult:
    """Tests for ScoringResult dataclass."""

    def test_scoring_result_creation(self):
        """Test creating a ScoringResult."""
        result = ScoringResult(
            trial_id="NCT12345678",
            composite_score=7.5,
            confidence=0.8,
            recommendation=Recommendation.STRONG_BUY,
            signal_count=5,
            total_weight=12,
            max_possible_weight=20,
            contributing_signals=[{"signal_type": "test"}],
        )

        assert result.trial_id == "NCT12345678"
        assert result.composite_score == 7.5
        assert result.confidence == 0.8
        assert result.recommendation == Recommendation.STRONG_BUY


class TestScoreNormalization:
    """Tests for score normalization."""

    @pytest.fixture
    def scorer(self):
        return SignalScorer()

    def test_positive_score_normalization(self, scorer):
        """Test that positive weights normalize to 5-10 range."""
        # Create mock signals with positive weight
        mock_signal = MagicMock()
        mock_signal.signal_type = "positive_signal"
        mock_signal.signal_weight = 5  # Max positive
        mock_signal.signal_value = "Test"
        mock_signal.detected_date = datetime.now()

        with patch("scoring.signal_scorer.TrialSignal") as mock_class:
            mock_class.get_by_trial.return_value = [mock_signal]

            result = scorer.score_trial("NCT12345678")

            assert result.composite_score >= 5.0
            assert result.composite_score <= 10.0

    def test_negative_score_normalization(self, scorer):
        """Test that negative weights normalize to 0-5 range."""
        mock_signal = MagicMock()
        mock_signal.signal_type = "negative_signal"
        mock_signal.signal_weight = -5  # Max negative
        mock_signal.signal_value = "Test"
        mock_signal.detected_date = datetime.now()

        with patch("scoring.signal_scorer.TrialSignal") as mock_class:
            mock_class.get_by_trial.return_value = [mock_signal]

            result = scorer.score_trial("NCT12345678")

            assert result.composite_score >= 0.0
            assert result.composite_score <= 5.0
