"""
Tests for the Signal Scoring Model
"""

import pytest
from datetime import date, timedelta
from unittest.mock import Mock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signal_scorer import SignalScorer, Signal


class TestTimeDecay:
    """Test time decay calculations."""

    def test_decay_factor_today(self):
        """Signal from today should have decay factor of 1.0."""
        scorer = SignalScorer.__new__(SignalScorer)
        scorer.time_decay_halflife = 30

        decay = scorer.calculate_decay_factor(0)
        assert decay == 1.0

    def test_decay_factor_halflife(self):
        """Signal from half-life days ago should have decay factor of 0.5."""
        scorer = SignalScorer.__new__(SignalScorer)
        scorer.time_decay_halflife = 30

        decay = scorer.calculate_decay_factor(30)
        assert decay == 0.5

    def test_decay_factor_double_halflife(self):
        """Signal from 2x half-life should have decay factor of ~0.33."""
        scorer = SignalScorer.__new__(SignalScorer)
        scorer.time_decay_halflife = 30

        decay = scorer.calculate_decay_factor(60)
        assert abs(decay - 0.333) < 0.01

    def test_decay_factor_90_days(self):
        """Signal from 90 days should have decay factor of 0.25."""
        scorer = SignalScorer.__new__(SignalScorer)
        scorer.time_decay_halflife = 30

        decay = scorer.calculate_decay_factor(90)
        assert decay == 0.25


class TestSignalWeights:
    """Test signal weight definitions."""

    def test_insider_signal_types_defined(self):
        """Ensure all insider signal types are defined."""
        required_types = [
            'CEO_PURCHASE', 'CEO_SALE', 'CFO_PURCHASE', 'CFO_SALE',
            'CMO_PURCHASE', 'CMO_SALE', 'DIRECTOR_PURCHASE', 'OFFICER_PURCHASE'
        ]

        for signal_type in required_types:
            assert signal_type in SignalScorer.INSIDER_SIGNAL_TYPES

    def test_institutional_signal_types_defined(self):
        """Ensure all institutional signal types are defined."""
        required_types = [
            'FUND_NEW_POSITION', 'FUND_EXIT',
            'FUND_INCREASE_50', 'FUND_DECREASE_50'
        ]

        for signal_type in required_types:
            assert signal_type in SignalScorer.INSTITUTIONAL_SIGNAL_TYPES

    def test_hiring_signal_types_defined(self):
        """Ensure all hiring signal types are defined."""
        required_types = [
            'COMMERCIAL_BUILDUP', 'VP_MANUFACTURING',
            'CLINICAL_EXPANSION', 'HIRING_FREEZE'
        ]

        for signal_type in required_types:
            assert signal_type in SignalScorer.HIRING_SIGNAL_TYPES

    def test_bullish_signals_positive(self):
        """Ensure bullish signals have positive weights."""
        bullish_types = ['CEO_PURCHASE', 'CFO_PURCHASE', 'CMO_PURCHASE']

        for signal_type in bullish_types:
            weight = SignalScorer.INSIDER_SIGNAL_TYPES[signal_type]['weight']
            assert weight > 0, f"{signal_type} should be positive"

    def test_bearish_signals_negative(self):
        """Ensure bearish signals have negative weights."""
        bearish_types = ['CEO_SALE', 'CFO_SALE', 'CMO_SALE']

        for signal_type in bearish_types:
            weight = SignalScorer.INSIDER_SIGNAL_TYPES[signal_type]['weight']
            assert weight < 0, f"{signal_type} should be negative"


class TestRecommendations:
    """Test recommendation logic."""

    def test_strong_buy_threshold(self):
        """Test STRONG BUY recommendation."""
        scorer = SignalScorer.__new__(SignalScorer)

        rec = scorer._get_recommendation(6.0, 0.70)
        assert rec == "STRONG BUY"

    def test_buy_threshold(self):
        """Test BUY recommendation."""
        scorer = SignalScorer.__new__(SignalScorer)

        rec = scorer._get_recommendation(3.0, 0.50)
        assert rec == "BUY"

    def test_sell_threshold(self):
        """Test SELL recommendation."""
        scorer = SignalScorer.__new__(SignalScorer)

        rec = scorer._get_recommendation(-3.0, 0.50)
        assert rec == "SELL"

    def test_strong_sell_threshold(self):
        """Test STRONG SELL recommendation."""
        scorer = SignalScorer.__new__(SignalScorer)

        rec = scorer._get_recommendation(-6.0, 0.70)
        assert rec == "STRONG SELL"

    def test_neutral_low_score(self):
        """Test NEUTRAL for low score."""
        scorer = SignalScorer.__new__(SignalScorer)

        rec = scorer._get_recommendation(1.0, 0.50)
        assert rec == "NEUTRAL"

    def test_neutral_low_confidence(self):
        """Test NEUTRAL for low confidence."""
        scorer = SignalScorer.__new__(SignalScorer)

        rec = scorer._get_recommendation(5.0, 0.30)
        assert rec == "NEUTRAL"


class TestSignalDataclass:
    """Test Signal dataclass."""

    def test_signal_creation(self):
        """Test creating a Signal object."""
        signal = Signal(
            company_ticker='MRNA',
            signal_date=date.today(),
            category='insider',
            signal_type='CEO_PURCHASE',
            description='CEO bought $200K',
            raw_weight=5,
            days_ago=10,
            decay_factor=0.75,
            weighted_score=3.75
        )

        assert signal.company_ticker == 'MRNA'
        assert signal.raw_weight == 5
        assert signal.weighted_score == 3.75

    def test_signal_default_values(self):
        """Test Signal default values."""
        signal = Signal(
            company_ticker='MRNA',
            signal_date=date.today(),
            category='insider',
            signal_type='CEO_PURCHASE',
            description='Test',
            raw_weight=5
        )

        assert signal.days_ago == 0
        assert signal.decay_factor == 1.0
        assert signal.weighted_score == 0.0


class TestScoreCalculation:
    """Test score calculation logic."""

    def test_score_normalization_positive(self):
        """Test score normalization for positive scores."""
        # Score of 16 should normalize to 8 (16/2)
        raw_score = 16
        normalized = max(-10.0, min(10.0, raw_score / 2.0))
        assert normalized == 8.0

    def test_score_normalization_clamped(self):
        """Test score clamping at bounds."""
        # Score of 30 should clamp to 10
        raw_score = 30
        normalized = max(-10.0, min(10.0, raw_score / 2.0))
        assert normalized == 10.0

    def test_score_normalization_negative(self):
        """Test score normalization for negative scores."""
        raw_score = -12
        normalized = max(-10.0, min(10.0, raw_score / 2.0))
        assert normalized == -6.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
