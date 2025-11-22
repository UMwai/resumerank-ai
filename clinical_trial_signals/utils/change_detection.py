"""
Change detection module for Clinical Trial Signal Detection System.

Compares current trial data with historical snapshots to detect significant changes.
"""
import logging
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config
from database.models import Trial, TrialHistory, TrialSignal

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes that can be detected."""
    STATUS_CHANGE = "status_change"
    ENROLLMENT_CHANGE = "enrollment_change"
    COMPLETION_DATE_CHANGE = "completion_date_change"
    ENDPOINT_CHANGE = "endpoint_change"
    SITES_CHANGE = "sites_change"


@dataclass
class DetectedChange:
    """Represents a detected change in trial data."""
    trial_id: str
    change_type: ChangeType
    old_value: Any
    new_value: Any
    signal_type: str
    signal_weight: int
    description: str
    raw_data: Dict[str, Any] = None


class ChangeDetector:
    """
    Detects significant changes in clinical trial data by comparing
    current state with historical snapshots.
    """

    # Status transitions and their signals
    STATUS_SIGNALS = {
        # Positive status changes
        ("RECRUITING", "ACTIVE_NOT_RECRUITING"): ("status_change_positive", 2, "Trial moved to active, recruitment complete"),
        ("ENROLLING_BY_INVITATION", "ACTIVE_NOT_RECRUITING"): ("status_change_positive", 2, "Trial moved to active phase"),
        ("NOT_YET_RECRUITING", "RECRUITING"): ("status_change_positive", 1, "Trial started recruiting"),

        # Negative status changes
        ("RECRUITING", "SUSPENDED"): ("status_change_negative", -3, "Trial suspended"),
        ("ACTIVE_NOT_RECRUITING", "SUSPENDED"): ("status_change_negative", -3, "Trial suspended"),
        ("RECRUITING", "TERMINATED"): ("status_change_negative", -4, "Trial terminated"),
        ("ACTIVE_NOT_RECRUITING", "TERMINATED"): ("status_change_negative", -4, "Trial terminated"),
        ("RECRUITING", "WITHDRAWN"): ("status_change_negative", -4, "Trial withdrawn"),
    }

    def __init__(self):
        self.weights = config.signal.weights

    def create_snapshot(self, trial: Trial) -> TrialHistory:
        """
        Create a historical snapshot of a trial's current state.

        Args:
            trial: Trial object to snapshot

        Returns:
            TrialHistory object (saved to database)
        """
        snapshot = TrialHistory(
            trial_id=trial.trial_id,
            enrollment_current=trial.enrollment_current,
            status=trial.status,
            expected_completion=trial.expected_completion,
            primary_endpoint=trial.primary_endpoint,
            raw_data=trial.raw_data
        )
        snapshot.save()
        logger.debug(f"Created snapshot for {trial.trial_id}")
        return snapshot

    def detect_changes(self, trial: Trial) -> List[DetectedChange]:
        """
        Compare current trial data with the most recent snapshot and detect changes.

        Args:
            trial: Current trial data

        Returns:
            List of DetectedChange objects
        """
        changes = []

        # Get previous snapshot
        previous = TrialHistory.get_latest(trial.trial_id)

        if not previous:
            # First time seeing this trial, create initial snapshot
            self.create_snapshot(trial)
            logger.info(f"Created initial snapshot for {trial.trial_id}")
            return changes

        # Check for status changes
        status_change = self._detect_status_change(trial, previous)
        if status_change:
            changes.append(status_change)

        # Check for enrollment changes
        enrollment_changes = self._detect_enrollment_changes(trial, previous)
        changes.extend(enrollment_changes)

        # Check for completion date changes
        date_change = self._detect_completion_date_change(trial, previous)
        if date_change:
            changes.append(date_change)

        # Check for endpoint changes
        endpoint_change = self._detect_endpoint_change(trial, previous)
        if endpoint_change:
            changes.append(endpoint_change)

        # Create new snapshot if any changes detected
        if changes:
            self.create_snapshot(trial)
            logger.info(f"Detected {len(changes)} changes for {trial.trial_id}")

        return changes

    def _detect_status_change(
        self,
        trial: Trial,
        previous: TrialHistory
    ) -> Optional[DetectedChange]:
        """Detect changes in trial status."""
        if not previous.status or trial.status == previous.status:
            return None

        old_status = previous.status
        new_status = trial.status

        # Check for known status transitions
        transition = (old_status, new_status)
        if transition in self.STATUS_SIGNALS:
            signal_type, weight, description = self.STATUS_SIGNALS[transition]
        else:
            # Generic status change
            if new_status in ("COMPLETED",):
                signal_type = "status_change_positive"
                weight = 2
                description = f"Trial status changed: {old_status} -> {new_status}"
            elif new_status in ("SUSPENDED", "TERMINATED", "WITHDRAWN"):
                signal_type = "status_change_negative"
                weight = -3
                description = f"Trial status changed: {old_status} -> {new_status}"
            else:
                signal_type = "status_change_neutral"
                weight = 0
                description = f"Trial status changed: {old_status} -> {new_status}"

        return DetectedChange(
            trial_id=trial.trial_id,
            change_type=ChangeType.STATUS_CHANGE,
            old_value=old_status,
            new_value=new_status,
            signal_type=signal_type,
            signal_weight=weight,
            description=description,
            raw_data={"old_status": old_status, "new_status": new_status}
        )

    def _detect_enrollment_changes(
        self,
        trial: Trial,
        previous: TrialHistory
    ) -> List[DetectedChange]:
        """Detect significant changes in enrollment."""
        changes = []

        if trial.enrollment_current is None or previous.enrollment_current is None:
            return changes

        old_enrollment = previous.enrollment_current
        new_enrollment = trial.enrollment_current

        if old_enrollment == new_enrollment:
            return changes

        # Calculate percentage change
        if old_enrollment > 0:
            pct_change = (new_enrollment - old_enrollment) / old_enrollment * 100
        else:
            pct_change = 100 if new_enrollment > 0 else 0

        # Significant increase (>10%)
        if pct_change >= 10:
            changes.append(DetectedChange(
                trial_id=trial.trial_id,
                change_type=ChangeType.ENROLLMENT_CHANGE,
                old_value=old_enrollment,
                new_value=new_enrollment,
                signal_type="enrollment_increase",
                signal_weight=self.weights.get("enrollment_increase", 2),
                description=f"Enrollment increased by {pct_change:.1f}% ({old_enrollment} -> {new_enrollment})",
                raw_data={"old": old_enrollment, "new": new_enrollment, "pct_change": pct_change}
            ))

        # Significant decrease (negative - very unusual and concerning)
        elif pct_change <= -5:
            changes.append(DetectedChange(
                trial_id=trial.trial_id,
                change_type=ChangeType.ENROLLMENT_CHANGE,
                old_value=old_enrollment,
                new_value=new_enrollment,
                signal_type="enrollment_decrease",
                signal_weight=self.weights.get("enrollment_decrease", -2),
                description=f"Enrollment decreased by {abs(pct_change):.1f}% ({old_enrollment} -> {new_enrollment})",
                raw_data={"old": old_enrollment, "new": new_enrollment, "pct_change": pct_change}
            ))

        # Check for early completion of enrollment
        if (trial.enrollment_target and new_enrollment >= trial.enrollment_target and
            old_enrollment < trial.enrollment_target):
            changes.append(DetectedChange(
                trial_id=trial.trial_id,
                change_type=ChangeType.ENROLLMENT_CHANGE,
                old_value=old_enrollment,
                new_value=new_enrollment,
                signal_type="early_enrollment",
                signal_weight=self.weights.get("early_enrollment", 3),
                description=f"Enrollment target reached ({new_enrollment}/{trial.enrollment_target})",
                raw_data={"current": new_enrollment, "target": trial.enrollment_target}
            ))

        return changes

    def _detect_completion_date_change(
        self,
        trial: Trial,
        previous: TrialHistory
    ) -> Optional[DetectedChange]:
        """Detect changes in expected completion date."""
        if not trial.expected_completion or not previous.expected_completion:
            return None

        if trial.expected_completion == previous.expected_completion:
            return None

        old_date = previous.expected_completion
        new_date = trial.expected_completion

        # Calculate difference in days
        date_diff = (new_date - old_date).days

        if abs(date_diff) < 30:  # Ignore small changes (<1 month)
            return None

        if date_diff < 0:  # Completion moved earlier
            return DetectedChange(
                trial_id=trial.trial_id,
                change_type=ChangeType.COMPLETION_DATE_CHANGE,
                old_value=old_date.isoformat(),
                new_value=new_date.isoformat(),
                signal_type="completion_date_accelerated",
                signal_weight=self.weights.get("completion_date_accelerated", 3),
                description=f"Completion date moved earlier by {abs(date_diff)} days ({old_date} -> {new_date})",
                raw_data={"old_date": old_date.isoformat(), "new_date": new_date.isoformat(), "days_diff": date_diff}
            )
        else:  # Completion delayed
            return DetectedChange(
                trial_id=trial.trial_id,
                change_type=ChangeType.COMPLETION_DATE_CHANGE,
                old_value=old_date.isoformat(),
                new_value=new_date.isoformat(),
                signal_type="completion_date_delayed",
                signal_weight=self.weights.get("completion_date_delayed", -3),
                description=f"Completion date delayed by {date_diff} days ({old_date} -> {new_date})",
                raw_data={"old_date": old_date.isoformat(), "new_date": new_date.isoformat(), "days_diff": date_diff}
            )

    def _detect_endpoint_change(
        self,
        trial: Trial,
        previous: TrialHistory
    ) -> Optional[DetectedChange]:
        """Detect changes in primary endpoint."""
        if not trial.primary_endpoint or not previous.primary_endpoint:
            return None

        old_endpoint = previous.primary_endpoint.strip().lower()
        new_endpoint = trial.primary_endpoint.strip().lower()

        if old_endpoint == new_endpoint:
            return None

        # Endpoint changed - this is typically a negative signal
        return DetectedChange(
            trial_id=trial.trial_id,
            change_type=ChangeType.ENDPOINT_CHANGE,
            old_value=previous.primary_endpoint,
            new_value=trial.primary_endpoint,
            signal_type="endpoint_change",
            signal_weight=self.weights.get("endpoint_change", -5),
            description=f"Primary endpoint modified",
            raw_data={"old_endpoint": previous.primary_endpoint, "new_endpoint": trial.primary_endpoint}
        )

    def process_trial_changes(self, trial: Trial) -> List[TrialSignal]:
        """
        Detect changes for a trial and create signal records.

        Args:
            trial: Trial to analyze

        Returns:
            List of created TrialSignal objects
        """
        changes = self.detect_changes(trial)
        signals = []

        for change in changes:
            signal = TrialSignal(
                trial_id=change.trial_id,
                signal_type=change.signal_type,
                signal_value=change.description,
                signal_weight=change.signal_weight,
                detected_date=datetime.now(),
                source="change_detection",
                raw_data=change.raw_data
            )
            signal.save()
            signals.append(signal)

        return signals

    def process_all_trials(self) -> Dict[str, List[TrialSignal]]:
        """
        Process all monitored trials for changes.

        Returns:
            Dictionary mapping trial_id to list of signals
        """
        trials = Trial.get_monitored()
        results = {}

        for trial in trials:
            signals = self.process_trial_changes(trial)
            if signals:
                results[trial.trial_id] = signals
                logger.info(f"Found {len(signals)} changes for {trial.trial_id}")

        return results


if __name__ == "__main__":
    # Test change detection
    logging.basicConfig(level=logging.INFO)

    print("Testing change detection module...")

    # Create mock trial data
    trial = Trial(
        trial_id="NCT12345678",
        company_ticker="TEST",
        drug_name="Test Drug",
        indication="Test Indication",
        phase="PHASE3",
        enrollment_target=500,
        enrollment_current=450,
        expected_completion=date(2025, 6, 1),
        status="RECRUITING",
        primary_endpoint="Overall survival"
    )

    detector = ChangeDetector()

    # First run - should create snapshot
    changes = detector.detect_changes(trial)
    print(f"First run: {len(changes)} changes detected")

    # Simulate changes
    trial.enrollment_current = 500
    trial.status = "ACTIVE_NOT_RECRUITING"
    trial.expected_completion = date(2025, 3, 1)

    # Second run - should detect changes
    changes = detector.detect_changes(trial)
    print(f"Second run: {len(changes)} changes detected")

    for change in changes:
        print(f"  - {change.signal_type}: {change.description} (weight: {change.signal_weight})")
