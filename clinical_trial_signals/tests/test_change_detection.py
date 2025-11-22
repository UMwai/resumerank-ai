"""
Tests for the change detection module.
"""
import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.change_detection import ChangeDetector, ChangeType, DetectedChange


class TestChangeDetector:
    """Tests for ChangeDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a ChangeDetector instance."""
        return ChangeDetector()

    @pytest.fixture
    def mock_trial(self):
        """Create a mock trial object."""
        trial = MagicMock()
        trial.trial_id = "NCT12345678"
        trial.enrollment_current = 250
        trial.enrollment_target = 500
        trial.status = "RECRUITING"
        trial.expected_completion = date(2025, 6, 1)
        trial.primary_endpoint = "Overall Survival"
        trial.raw_data = {}
        return trial

    @pytest.fixture
    def mock_history(self):
        """Create a mock trial history object."""
        history = MagicMock()
        history.enrollment_current = 200
        history.status = "RECRUITING"
        history.expected_completion = date(2025, 6, 1)
        history.primary_endpoint = "Overall Survival"
        return history

    def test_detect_enrollment_increase(self, detector, mock_trial, mock_history):
        """Test detection of enrollment increase."""
        mock_history.enrollment_current = 200
        mock_trial.enrollment_current = 250  # 25% increase

        with patch("utils.change_detection.TrialHistory") as mock_class:
            mock_class.get_latest.return_value = mock_history

            changes = detector._detect_enrollment_changes(mock_trial, mock_history)

            assert len(changes) == 1
            assert changes[0].signal_type == "enrollment_increase"
            assert changes[0].signal_weight > 0

    def test_detect_enrollment_decrease(self, detector, mock_trial, mock_history):
        """Test detection of enrollment decrease."""
        mock_history.enrollment_current = 300
        mock_trial.enrollment_current = 250  # Decrease

        changes = detector._detect_enrollment_changes(mock_trial, mock_history)

        assert len(changes) == 1
        assert changes[0].signal_type == "enrollment_decrease"
        assert changes[0].signal_weight < 0

    def test_detect_early_enrollment(self, detector, mock_trial, mock_history):
        """Test detection of early enrollment completion."""
        mock_history.enrollment_current = 480
        mock_trial.enrollment_current = 510  # Above target of 500
        mock_trial.enrollment_target = 500

        changes = detector._detect_enrollment_changes(mock_trial, mock_history)

        # Should detect both increase and early enrollment
        assert any(c.signal_type == "early_enrollment" for c in changes)

    def test_detect_status_change_positive(self, detector, mock_trial, mock_history):
        """Test detection of positive status change."""
        mock_history.status = "RECRUITING"
        mock_trial.status = "ACTIVE_NOT_RECRUITING"

        change = detector._detect_status_change(mock_trial, mock_history)

        assert change is not None
        assert change.signal_type == "status_change_positive"
        assert change.signal_weight > 0

    def test_detect_status_change_negative(self, detector, mock_trial, mock_history):
        """Test detection of negative status change."""
        mock_history.status = "RECRUITING"
        mock_trial.status = "SUSPENDED"

        change = detector._detect_status_change(mock_trial, mock_history)

        assert change is not None
        assert change.signal_type == "status_change_negative"
        assert change.signal_weight < 0

    def test_detect_completion_date_accelerated(self, detector, mock_trial, mock_history):
        """Test detection of accelerated completion date."""
        mock_history.expected_completion = date(2025, 6, 1)
        mock_trial.expected_completion = date(2025, 3, 1)  # 3 months earlier

        change = detector._detect_completion_date_change(mock_trial, mock_history)

        assert change is not None
        assert change.signal_type == "completion_date_accelerated"
        assert change.signal_weight > 0

    def test_detect_completion_date_delayed(self, detector, mock_trial, mock_history):
        """Test detection of delayed completion date."""
        mock_history.expected_completion = date(2025, 6, 1)
        mock_trial.expected_completion = date(2025, 12, 1)  # 6 months delay

        change = detector._detect_completion_date_change(mock_trial, mock_history)

        assert change is not None
        assert change.signal_type == "completion_date_delayed"
        assert change.signal_weight < 0

    def test_ignore_small_date_change(self, detector, mock_trial, mock_history):
        """Test that small date changes (<30 days) are ignored."""
        mock_history.expected_completion = date(2025, 6, 1)
        mock_trial.expected_completion = date(2025, 6, 15)  # Only 14 days

        change = detector._detect_completion_date_change(mock_trial, mock_history)

        assert change is None

    def test_detect_endpoint_change(self, detector, mock_trial, mock_history):
        """Test detection of endpoint change."""
        mock_history.primary_endpoint = "Overall Survival"
        mock_trial.primary_endpoint = "Progression Free Survival"

        change = detector._detect_endpoint_change(mock_trial, mock_history)

        assert change is not None
        assert change.signal_type == "endpoint_change"
        assert change.signal_weight < 0  # Endpoint change is negative

    def test_no_endpoint_change_when_same(self, detector, mock_trial, mock_history):
        """Test no change detected when endpoint is same."""
        mock_history.primary_endpoint = "Overall Survival"
        mock_trial.primary_endpoint = "overall survival"  # Same, different case

        change = detector._detect_endpoint_change(mock_trial, mock_history)

        assert change is None

    def test_detect_changes_creates_snapshot(self, detector, mock_trial):
        """Test that detect_changes creates snapshot on first run."""
        with patch("utils.change_detection.TrialHistory") as mock_class:
            mock_class.get_latest.return_value = None  # No history

            with patch.object(detector, "create_snapshot") as mock_snapshot:
                changes = detector.detect_changes(mock_trial)

                mock_snapshot.assert_called_once()
                assert len(changes) == 0


class TestDetectedChange:
    """Tests for DetectedChange dataclass."""

    def test_detected_change_creation(self):
        """Test creating a DetectedChange."""
        change = DetectedChange(
            trial_id="NCT12345678",
            change_type=ChangeType.ENROLLMENT_CHANGE,
            old_value=200,
            new_value=250,
            signal_type="enrollment_increase",
            signal_weight=2,
            description="Enrollment increased by 25%",
        )

        assert change.trial_id == "NCT12345678"
        assert change.change_type == ChangeType.ENROLLMENT_CHANGE
        assert change.old_value == 200
        assert change.new_value == 250


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_change_type_values(self):
        """Test ChangeType enum values."""
        assert ChangeType.STATUS_CHANGE.value == "status_change"
        assert ChangeType.ENROLLMENT_CHANGE.value == "enrollment_change"
        assert ChangeType.COMPLETION_DATE_CHANGE.value == "completion_date_change"
        assert ChangeType.ENDPOINT_CHANGE.value == "endpoint_change"
