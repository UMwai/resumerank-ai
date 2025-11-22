"""
Tests for Accuracy Tracker Module

Tests cover:
- Prediction dataclass
- AccuracyMetrics dataclass
- AccuracyTracker operations
- Performance metrics calculation
"""

import os
import pytest
import tempfile
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.accuracy_tracker import (
    Prediction,
    AccuracyMetrics,
    AccuracyTracker,
    generate_demo_predictions,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tracker(temp_db):
    """Create an AccuracyTracker with temporary database."""
    return AccuracyTracker(db_path=temp_db)


class TestPrediction:
    """Test Prediction dataclass."""

    def test_prediction_creation(self):
        """Test creating a Prediction."""
        pred = Prediction(
            signal_id="test_1",
            source="clinical_trial",
            ticker="MRNA",
            company_name="Moderna",
            signal_type="bullish",
            predicted_direction="up",
            score=0.85,
            confidence=0.90,
            entry_price=150.0,
        )

        assert pred.signal_id == "test_1"
        assert pred.source == "clinical_trial"
        assert pred.ticker == "MRNA"
        assert pred.predicted_direction == "up"
        assert pred.score == 0.85

    def test_prediction_to_dict(self):
        """Test Prediction to_dict conversion."""
        pred = Prediction(
            id=1,
            signal_id="test_2",
            source="patent",
            ticker="PFE",
            score=0.7,
            confidence=0.8,
        )

        data = pred.to_dict()

        assert data["id"] == 1
        assert data["signal_id"] == "test_2"
        assert data["source"] == "patent"
        assert data["ticker"] == "PFE"


class TestAccuracyMetrics:
    """Test AccuracyMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating AccuracyMetrics."""
        metrics = AccuracyMetrics(
            total_predictions=100,
            evaluated_predictions=80,
            wins=50,
            losses=20,
            breakeven=10,
            pending=20,
        )

        assert metrics.total_predictions == 100
        assert metrics.wins == 50
        assert metrics.losses == 20

    def test_metrics_to_dict(self):
        """Test AccuracyMetrics to_dict conversion."""
        metrics = AccuracyMetrics(
            total_predictions=50,
            wins=30,
            losses=15,
            win_rate=0.66,
        )

        data = metrics.to_dict()

        assert data["total_predictions"] == 50
        assert data["wins"] == 30
        assert data["win_rate"] == 0.66


class TestAccuracyTracker:
    """Test AccuracyTracker operations."""

    def test_track_prediction(self, tracker):
        """Test tracking a new prediction."""
        pred = tracker.track_prediction(
            signal_id="sig_001",
            source="clinical_trial",
            ticker="MRNA",
            signal_type="bullish",
            score=0.85,
            confidence=0.90,
            entry_price=150.0,
            target_price=175.0,
            stop_price=140.0,
            company_name="Moderna Inc.",
        )

        assert pred is not None
        assert pred.id is not None
        assert pred.signal_id == "sig_001"
        assert pred.ticker == "MRNA"
        assert pred.outcome == "pending"

    def test_track_duplicate_prediction(self, tracker):
        """Test tracking duplicate prediction returns None."""
        tracker.track_prediction(
            signal_id="sig_dup",
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.8,
        )

        duplicate = tracker.track_prediction(
            signal_id="sig_dup",  # Same ID
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.8,
        )

        assert duplicate is None

    def test_record_outcome_win(self, tracker):
        """Test recording a winning outcome."""
        pred = tracker.track_prediction(
            signal_id="sig_win",
            source="insider",
            ticker="GILD",
            signal_type="bullish",
            score=0.8,
            confidence=0.85,
            entry_price=70.0,
        )

        success = tracker.record_outcome(
            prediction_id=pred.id,
            outcome="win",
            exit_price=80.0,
            notes="Price target hit",
        )

        assert success is True

    def test_record_outcome_loss(self, tracker):
        """Test recording a losing outcome."""
        pred = tracker.track_prediction(
            signal_id="sig_loss",
            source="clinical_trial",
            ticker="ABBV",
            signal_type="bullish",
            score=0.75,
            confidence=0.8,
            entry_price=150.0,
        )

        success = tracker.record_outcome(
            prediction_id=pred.id,
            outcome="loss",
            exit_price=130.0,
            notes="Stop loss triggered",
        )

        assert success is True


class TestAccuracyMetricsCalculation:
    """Test accuracy metrics calculation."""

    def test_get_accuracy_metrics_empty(self, tracker):
        """Test getting metrics with no data."""
        metrics = tracker.get_accuracy_metrics()

        assert metrics.total_predictions == 0
        assert metrics.win_rate == 0.0

    def test_get_accuracy_metrics_with_data(self, tracker):
        """Test getting metrics with predictions."""
        # Add some predictions with outcomes
        for i in range(5):
            pred = tracker.track_prediction(
                signal_id=f"sig_metric_{i}",
                source="clinical_trial",
                ticker="MRNA",
                signal_type="bullish",
                score=0.8,
                confidence=0.85,
                entry_price=100.0,
            )
            outcome = "win" if i < 3 else "loss"
            tracker.record_outcome(pred.id, outcome, exit_price=110.0 if outcome == "win" else 90.0)

        metrics = tracker.get_accuracy_metrics()

        assert metrics.total_predictions == 5
        assert metrics.evaluated_predictions == 5
        assert metrics.wins == 3
        assert metrics.losses == 2
        assert metrics.win_rate == 0.6

    def test_get_accuracy_metrics_by_source(self, tracker):
        """Test getting metrics filtered by source."""
        # Add clinical trial prediction
        pred1 = tracker.track_prediction(
            signal_id="sig_source_1",
            source="clinical_trial",
            ticker="MRNA",
            signal_type="bullish",
            score=0.8,
            confidence=0.85,
        )
        tracker.record_outcome(pred1.id, "win")

        # Add patent prediction
        pred2 = tracker.track_prediction(
            signal_id="sig_source_2",
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.8,
        )
        tracker.record_outcome(pred2.id, "loss")

        clinical_metrics = tracker.get_accuracy_metrics(source="clinical_trial")
        patent_metrics = tracker.get_accuracy_metrics(source="patent")

        assert clinical_metrics.total_predictions == 1
        assert clinical_metrics.wins == 1

        assert patent_metrics.total_predictions == 1
        assert patent_metrics.losses == 1

    def test_get_accuracy_metrics_by_ticker(self, tracker):
        """Test getting metrics filtered by ticker."""
        pred = tracker.track_prediction(
            signal_id="sig_ticker",
            source="insider",
            ticker="GILD",
            signal_type="bullish",
            score=0.8,
            confidence=0.85,
        )
        tracker.record_outcome(pred.id, "win")

        gild_metrics = tracker.get_accuracy_metrics(ticker="GILD")
        mrna_metrics = tracker.get_accuracy_metrics(ticker="MRNA")

        assert gild_metrics.total_predictions == 1
        assert mrna_metrics.total_predictions == 0

    def test_get_metrics_by_source(self, tracker):
        """Test getting metrics grouped by source."""
        # Add predictions for each source
        for source in ["clinical_trial", "patent", "insider"]:
            pred = tracker.track_prediction(
                signal_id=f"sig_{source}",
                source=source,
                ticker="MRNA",
                signal_type="bullish",
                score=0.8,
                confidence=0.85,
            )
            tracker.record_outcome(pred.id, "win")

        metrics_by_source = tracker.get_metrics_by_source()

        assert "clinical_trial" in metrics_by_source
        assert "patent" in metrics_by_source
        assert "insider" in metrics_by_source

    def test_roi_calculation(self, tracker):
        """Test ROI calculation for bullish prediction."""
        pred = tracker.track_prediction(
            signal_id="sig_roi_bull",
            source="clinical_trial",
            ticker="MRNA",
            signal_type="bullish",
            score=0.85,
            confidence=0.90,
            entry_price=100.0,
        )

        # Price went up 10%
        tracker.record_outcome(pred.id, "win", exit_price=110.0)

        metrics = tracker.get_accuracy_metrics()
        # ROI should be positive for bullish winning trade
        assert metrics.total_roi > 0

    def test_roi_calculation_bearish(self, tracker):
        """Test ROI calculation for bearish prediction."""
        pred = tracker.track_prediction(
            signal_id="sig_roi_bear",
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.85,
            entry_price=50.0,
        )

        # Price went down 10% (bearish was correct)
        tracker.record_outcome(pred.id, "win", exit_price=45.0)

        metrics = tracker.get_accuracy_metrics()
        # ROI should be positive for bearish winning trade
        assert metrics.total_roi > 0


class TestAccuracyTrackerAdvanced:
    """Test advanced accuracy tracker features."""

    def test_get_roi_by_signal_type(self, tracker):
        """Test getting ROI breakdown by signal type."""
        # Add bullish prediction
        pred1 = tracker.track_prediction(
            signal_id="sig_type_1",
            source="clinical_trial",
            ticker="MRNA",
            signal_type="bullish",
            score=0.8,
            confidence=0.85,
            entry_price=100.0,
        )
        tracker.record_outcome(pred1.id, "win", exit_price=110.0)

        # Add bearish prediction
        pred2 = tracker.track_prediction(
            signal_id="sig_type_2",
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.8,
            entry_price=50.0,
        )
        tracker.record_outcome(pred2.id, "win", exit_price=45.0)

        roi_by_type = tracker.get_roi_by_signal_type()

        assert "bullish" in roi_by_type
        assert "bearish" in roi_by_type

    def test_get_predictions_df(self, tracker):
        """Test getting predictions as DataFrame."""
        tracker.track_prediction(
            signal_id="sig_df_1",
            source="clinical_trial",
            ticker="MRNA",
            signal_type="bullish",
            score=0.8,
            confidence=0.85,
        )
        tracker.track_prediction(
            signal_id="sig_df_2",
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.8,
        )

        df = tracker.get_predictions_df(limit=10)

        assert len(df) == 2
        assert "ticker" in df.columns
        assert "score" in df.columns

    def test_confidence_accuracy_breakdown(self, tracker):
        """Test accuracy breakdown by confidence level."""
        # High confidence prediction
        pred_high = tracker.track_prediction(
            signal_id="sig_conf_high",
            source="clinical_trial",
            ticker="MRNA",
            signal_type="bullish",
            score=0.85,
            confidence=0.90,  # High confidence
        )
        tracker.record_outcome(pred_high.id, "win")

        # Low confidence prediction
        pred_low = tracker.track_prediction(
            signal_id="sig_conf_low",
            source="patent",
            ticker="PFE",
            signal_type="bearish",
            score=0.3,
            confidence=0.40,  # Low confidence
        )
        tracker.record_outcome(pred_low.id, "loss")

        metrics = tracker.get_accuracy_metrics()

        # High confidence accuracy should be 100% (1 win out of 1)
        assert metrics.high_confidence_accuracy == 1.0
        # Low confidence accuracy should be 0% (0 wins out of 1)
        assert metrics.low_confidence_accuracy == 0.0


class TestDemoPredictions:
    """Test demo prediction generation."""

    def test_generate_demo_predictions(self, tracker):
        """Test generating demo predictions."""
        generate_demo_predictions(tracker, count=20)

        metrics = tracker.get_accuracy_metrics()

        # Should have approximately 20 predictions
        assert metrics.total_predictions >= 15  # Some may be duplicates
        # Most should have outcomes (80%)
        assert metrics.evaluated_predictions > 10
