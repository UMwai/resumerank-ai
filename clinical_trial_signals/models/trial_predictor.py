"""
Trial Outcome Prediction Model for Clinical Trial Signal Detection System.

Predicts probability of clinical trial success (0-100%) based on:
- Enrollment rate and timeline adherence
- Protocol changes and amendments
- Insider trading activity
- Historical signal patterns
- Trial phase and indication

Uses a weighted feature scoring model trained on historical Phase 3 trial outcomes.
"""
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import config

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Trial risk level classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TrialPhase(Enum):
    """Clinical trial phases."""
    PHASE1 = "PHASE1"
    PHASE1_2 = "PHASE1/PHASE2"
    PHASE2 = "PHASE2"
    PHASE2_3 = "PHASE2/PHASE3"
    PHASE3 = "PHASE3"
    PHASE4 = "PHASE4"


@dataclass
class TrialFeatures:
    """Features extracted for prediction model."""
    # Enrollment features
    enrollment_rate: float  # 0-2, where 1.0 = on track
    enrollment_completion_pct: float  # 0-100
    days_ahead_behind_enrollment: int  # Positive = ahead, negative = behind

    # Timeline features
    timeline_adherence: float  # 0-1, where 1.0 = perfectly on track
    completion_date_changes: int  # Number of date changes
    days_to_completion: int  # Days until expected completion

    # Protocol features
    protocol_amendments: int  # Number of protocol changes
    endpoint_changes: int  # Number of primary endpoint changes
    site_changes: int  # Net change in trial sites (positive = added)

    # Activity features
    insider_buy_signals: int  # Count of insider buying signals
    insider_sell_signals: int  # Count of insider selling signals
    sec_positive_signals: int  # Positive SEC filing signals
    sec_negative_signals: int  # Negative SEC filing signals

    # Historical features
    sponsor_success_rate: float  # Historical success rate of sponsor (0-1)
    indication_success_rate: float  # Historical success rate for indication (0-1)
    phase_base_success_rate: float  # Base success rate for trial phase (0-1)

    # Signal features
    total_signal_count: int  # Total signals detected
    net_signal_weight: int  # Sum of all signal weights
    signal_consistency: float  # 0-1, higher = more consistent direction


@dataclass
class PredictionResult:
    """Result of trial outcome prediction."""
    trial_id: str
    success_probability: float  # 0-100%
    confidence_interval: Tuple[float, float]  # (low, high) bounds
    risk_level: RiskLevel
    key_factors: List[Dict[str, Any]]  # Factors influencing prediction
    feature_contributions: Dict[str, float]  # Feature name -> contribution
    prediction_date: date
    model_version: str
    warnings: List[str]


class TrialPredictor:
    """
    Predicts clinical trial success probability using a weighted feature model.

    The model uses historical success rates and real-time signals to estimate
    the probability of a Phase 3 trial meeting its primary endpoints.
    """

    # Model version for tracking
    MODEL_VERSION = "1.0.0"

    # Base success rates by phase (from literature)
    PHASE_SUCCESS_RATES = {
        TrialPhase.PHASE1: 0.66,
        TrialPhase.PHASE1_2: 0.52,
        TrialPhase.PHASE2: 0.31,
        TrialPhase.PHASE2_3: 0.35,
        TrialPhase.PHASE3: 0.58,
        TrialPhase.PHASE4: 0.85,
    }

    # Success rates by indication (from historical data)
    INDICATION_SUCCESS_RATES = {
        "oncology": 0.40,
        "hematology": 0.55,
        "neurology": 0.35,
        "cardiovascular": 0.50,
        "immunology": 0.45,
        "infectious_disease": 0.60,
        "rare_disease": 0.65,
        "metabolic": 0.55,
        "respiratory": 0.50,
        "default": 0.50,
    }

    # Feature weights for prediction model
    FEATURE_WEIGHTS = {
        # Enrollment features (high importance)
        "enrollment_rate": 0.15,
        "enrollment_completion_pct": 0.08,
        "days_ahead_behind_enrollment": 0.07,

        # Timeline features (medium importance)
        "timeline_adherence": 0.10,
        "completion_date_changes": 0.06,

        # Protocol features (high importance - negative signals)
        "protocol_amendments": 0.08,
        "endpoint_changes": 0.12,  # Very high weight - endpoint changes are critical
        "site_changes": 0.05,

        # Activity features (medium importance)
        "insider_activity": 0.08,
        "sec_signals": 0.06,

        # Signal features
        "signal_consistency": 0.08,
        "net_signal_weight": 0.07,
    }

    def __init__(self, db_connection=None):
        """
        Initialize the trial predictor.

        Args:
            db_connection: Optional database connection for loading historical data
        """
        self.db = db_connection
        self._sponsor_cache: Dict[str, float] = {}

    def extract_features(
        self,
        trial_data: Dict[str, Any],
        signals: List[Dict[str, Any]],
        history: Optional[List[Dict]] = None
    ) -> TrialFeatures:
        """
        Extract features from trial data and signals.

        Args:
            trial_data: Trial information dict
            signals: List of detected signals
            history: Optional historical snapshots

        Returns:
            TrialFeatures object
        """
        # Extract enrollment features
        enrollment_target = trial_data.get("enrollment_target", 0) or 0
        enrollment_current = trial_data.get("enrollment_current", 0) or 0

        enrollment_completion_pct = (
            (enrollment_current / enrollment_target * 100) if enrollment_target > 0 else 0
        )

        # Calculate enrollment rate relative to timeline
        start_date = trial_data.get("start_date")
        expected_completion = trial_data.get("expected_completion")

        if start_date and expected_completion and enrollment_target > 0:
            total_days = (expected_completion - start_date).days
            elapsed_days = (date.today() - start_date).days
            expected_progress = (elapsed_days / total_days) if total_days > 0 else 0
            actual_progress = enrollment_current / enrollment_target
            enrollment_rate = actual_progress / expected_progress if expected_progress > 0 else 1.0
            enrollment_rate = min(2.0, max(0.0, enrollment_rate))

            # Days ahead/behind
            expected_enrolled = int(enrollment_target * expected_progress)
            enrollment_diff = enrollment_current - expected_enrolled
            # Convert to approximate days
            daily_target = enrollment_target / total_days if total_days > 0 else 1
            days_ahead_behind = int(enrollment_diff / daily_target) if daily_target > 0 else 0
        else:
            enrollment_rate = 1.0
            days_ahead_behind = 0

        # Extract timeline features
        days_to_completion = (
            (expected_completion - date.today()).days
            if expected_completion else 365
        )

        # Count completion date changes from history
        completion_date_changes = 0
        if history:
            prev_completion = None
            for snapshot in history:
                curr_completion = snapshot.get("expected_completion")
                if prev_completion and curr_completion != prev_completion:
                    completion_date_changes += 1
                prev_completion = curr_completion

        timeline_adherence = self._calculate_timeline_adherence(
            trial_data, completion_date_changes
        )

        # Extract protocol features from signals
        protocol_amendments = sum(
            1 for s in signals if "protocol" in s.get("signal_type", "").lower()
        )
        endpoint_changes = sum(
            1 for s in signals if "endpoint" in s.get("signal_type", "").lower()
        )
        site_changes = sum(
            s.get("signal_weight", 0) for s in signals
            if "site" in s.get("signal_type", "").lower()
        )

        # Extract insider activity
        insider_buy = sum(
            1 for s in signals if s.get("signal_type") == "insider_buying"
        )
        insider_sell = sum(
            1 for s in signals if s.get("signal_type") == "insider_selling"
        )

        # Extract SEC signals
        sec_positive = sum(
            1 for s in signals if "sec" in s.get("signal_type", "").lower()
            and s.get("signal_weight", 0) > 0
        )
        sec_negative = sum(
            1 for s in signals if "sec" in s.get("signal_type", "").lower()
            and s.get("signal_weight", 0) < 0
        )

        # Get historical success rates
        sponsor = trial_data.get("sponsor", "")
        sponsor_success_rate = self._get_sponsor_success_rate(sponsor)

        indication = trial_data.get("indication", "").lower()
        indication_success_rate = self._get_indication_success_rate(indication)

        phase_str = trial_data.get("phase", "PHASE3")
        phase_base_rate = self._get_phase_success_rate(phase_str)

        # Calculate signal metrics
        total_signals = len(signals)
        net_weight = sum(s.get("signal_weight", 0) for s in signals)
        signal_consistency = self._calculate_signal_consistency(signals)

        return TrialFeatures(
            enrollment_rate=enrollment_rate,
            enrollment_completion_pct=enrollment_completion_pct,
            days_ahead_behind_enrollment=days_ahead_behind,
            timeline_adherence=timeline_adherence,
            completion_date_changes=completion_date_changes,
            days_to_completion=days_to_completion,
            protocol_amendments=protocol_amendments,
            endpoint_changes=endpoint_changes,
            site_changes=site_changes,
            insider_buy_signals=insider_buy,
            insider_sell_signals=insider_sell,
            sec_positive_signals=sec_positive,
            sec_negative_signals=sec_negative,
            sponsor_success_rate=sponsor_success_rate,
            indication_success_rate=indication_success_rate,
            phase_base_success_rate=phase_base_rate,
            total_signal_count=total_signals,
            net_signal_weight=net_weight,
            signal_consistency=signal_consistency,
        )

    def _calculate_timeline_adherence(
        self,
        trial_data: Dict,
        date_changes: int
    ) -> float:
        """Calculate timeline adherence score."""
        # Start with perfect adherence
        adherence = 1.0

        # Penalize for each completion date change
        adherence -= date_changes * 0.15

        # Check if trial is delayed
        expected = trial_data.get("primary_completion_date") or trial_data.get("expected_completion")
        if expected and expected < date.today():
            # Trial is past expected completion
            days_overdue = (date.today() - expected).days
            adherence -= min(0.3, days_overdue / 365 * 0.5)

        return max(0.0, min(1.0, adherence))

    def _calculate_signal_consistency(self, signals: List[Dict]) -> float:
        """Calculate how consistent signals are in direction."""
        if not signals:
            return 0.5

        positive = sum(1 for s in signals if s.get("signal_weight", 0) > 0)
        negative = sum(1 for s in signals if s.get("signal_weight", 0) < 0)
        total = positive + negative

        if total == 0:
            return 0.5

        # Higher = more consistent
        dominant = max(positive, negative)
        return dominant / total

    def _get_sponsor_success_rate(self, sponsor: str) -> float:
        """Get historical success rate for sponsor."""
        if not sponsor:
            return 0.5

        # Check cache
        if sponsor in self._sponsor_cache:
            return self._sponsor_cache[sponsor]

        # Default rates for major sponsors (placeholder)
        major_sponsors = {
            "pfizer": 0.62,
            "roche": 0.60,
            "novartis": 0.58,
            "merck": 0.59,
            "johnson": 0.57,
            "abbvie": 0.61,
            "bristol": 0.58,
            "astrazeneca": 0.56,
            "gilead": 0.60,
            "amgen": 0.59,
        }

        sponsor_lower = sponsor.lower()
        for key, rate in major_sponsors.items():
            if key in sponsor_lower:
                self._sponsor_cache[sponsor] = rate
                return rate

        # Default for unknown sponsors
        self._sponsor_cache[sponsor] = 0.50
        return 0.50

    def _get_indication_success_rate(self, indication: str) -> float:
        """Get historical success rate for indication."""
        if not indication:
            return self.INDICATION_SUCCESS_RATES["default"]

        indication_lower = indication.lower()

        # Check for keywords
        for key, rate in self.INDICATION_SUCCESS_RATES.items():
            if key != "default" and key in indication_lower:
                return rate

        # Check for cancer/tumor keywords
        cancer_keywords = ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia", "melanoma"]
        if any(kw in indication_lower for kw in cancer_keywords):
            return self.INDICATION_SUCCESS_RATES["oncology"]

        return self.INDICATION_SUCCESS_RATES["default"]

    def _get_phase_success_rate(self, phase_str: str) -> float:
        """Get base success rate for trial phase."""
        try:
            phase = TrialPhase(phase_str.upper())
            return self.PHASE_SUCCESS_RATES.get(phase, 0.50)
        except (ValueError, AttributeError):
            # Try to extract phase number
            if "3" in str(phase_str):
                return self.PHASE_SUCCESS_RATES[TrialPhase.PHASE3]
            elif "2" in str(phase_str):
                return self.PHASE_SUCCESS_RATES[TrialPhase.PHASE2]
            return 0.50

    def predict(
        self,
        trial_data: Dict[str, Any],
        signals: List[Dict[str, Any]],
        history: Optional[List[Dict]] = None
    ) -> PredictionResult:
        """
        Predict trial success probability.

        Args:
            trial_data: Trial information dictionary
            signals: List of detected signals
            history: Optional historical snapshots

        Returns:
            PredictionResult with probability and analysis
        """
        trial_id = trial_data.get("trial_id", "unknown")
        logger.info(f"Predicting outcome for trial {trial_id}")

        # Extract features
        features = self.extract_features(trial_data, signals, history)

        # Calculate base probability from historical rates
        base_prob = self._calculate_base_probability(features)

        # Calculate feature contributions
        contributions = self._calculate_feature_contributions(features)

        # Calculate adjustment from features
        adjustment = sum(contributions.values())

        # Combine base probability with adjustments
        # Use logit transformation for better probability handling
        base_logit = self._prob_to_logit(base_prob)
        adjusted_logit = base_logit + adjustment
        success_probability = self._logit_to_prob(adjusted_logit)

        # Ensure valid range
        success_probability = max(1.0, min(99.0, success_probability * 100))

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(
            success_probability, features
        )

        # Determine risk level
        risk_level = self._assess_risk_level(success_probability, features)

        # Identify key factors
        key_factors = self._identify_key_factors(features, contributions)

        # Generate warnings
        warnings = self._generate_warnings(features, success_probability)

        return PredictionResult(
            trial_id=trial_id,
            success_probability=round(success_probability, 1),
            confidence_interval=confidence_interval,
            risk_level=risk_level,
            key_factors=key_factors,
            feature_contributions={k: round(v, 3) for k, v in contributions.items()},
            prediction_date=date.today(),
            model_version=self.MODEL_VERSION,
            warnings=warnings,
        )

    def _calculate_base_probability(self, features: TrialFeatures) -> float:
        """Calculate base probability from historical rates."""
        # Weighted average of historical rates
        weights = {"phase": 0.4, "indication": 0.3, "sponsor": 0.3}

        base_prob = (
            features.phase_base_success_rate * weights["phase"] +
            features.indication_success_rate * weights["indication"] +
            features.sponsor_success_rate * weights["sponsor"]
        )

        return base_prob

    def _calculate_feature_contributions(self, features: TrialFeatures) -> Dict[str, float]:
        """Calculate contribution of each feature to prediction."""
        contributions = {}

        # Enrollment rate contribution
        # >1.0 = positive, <1.0 = negative
        enrollment_adj = (features.enrollment_rate - 1.0) * 0.5
        contributions["enrollment_rate"] = enrollment_adj * self.FEATURE_WEIGHTS["enrollment_rate"]

        # Enrollment completion
        if features.enrollment_completion_pct >= 90:
            contributions["enrollment_completion"] = 0.1
        elif features.enrollment_completion_pct >= 70:
            contributions["enrollment_completion"] = 0.05
        elif features.enrollment_completion_pct < 30:
            contributions["enrollment_completion"] = -0.05
        else:
            contributions["enrollment_completion"] = 0.0

        # Timeline adherence
        contributions["timeline_adherence"] = (
            (features.timeline_adherence - 0.5) * self.FEATURE_WEIGHTS["timeline_adherence"]
        )

        # Completion date changes (negative impact)
        contributions["completion_date_changes"] = (
            -features.completion_date_changes * 0.08
        )

        # Endpoint changes (major negative impact)
        contributions["endpoint_changes"] = (
            -features.endpoint_changes * 0.25
        )

        # Protocol amendments (minor negative impact)
        contributions["protocol_amendments"] = (
            -features.protocol_amendments * 0.05
        )

        # Site changes (positive if adding)
        if features.site_changes > 0:
            contributions["site_changes"] = min(0.1, features.site_changes * 0.02)
        else:
            contributions["site_changes"] = max(-0.1, features.site_changes * 0.03)

        # Insider activity
        insider_net = features.insider_buy_signals - features.insider_sell_signals
        contributions["insider_activity"] = insider_net * 0.05

        # SEC signals
        sec_net = features.sec_positive_signals - features.sec_negative_signals
        contributions["sec_signals"] = sec_net * 0.04

        # Signal consistency
        if features.signal_consistency > 0.8 and features.net_signal_weight > 0:
            contributions["signal_momentum"] = 0.08
        elif features.signal_consistency > 0.8 and features.net_signal_weight < 0:
            contributions["signal_momentum"] = -0.08
        else:
            contributions["signal_momentum"] = 0.0

        return contributions

    def _prob_to_logit(self, prob: float) -> float:
        """Convert probability to logit (log-odds)."""
        prob = max(0.01, min(0.99, prob))
        return math.log(prob / (1 - prob))

    def _logit_to_prob(self, logit: float) -> float:
        """Convert logit to probability."""
        return 1 / (1 + math.exp(-logit))

    def _calculate_confidence_interval(
        self,
        probability: float,
        features: TrialFeatures
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        # Base uncertainty
        base_uncertainty = 15.0

        # Reduce uncertainty with more signals
        signal_reduction = min(5.0, features.total_signal_count * 0.5)

        # Reduce uncertainty with higher enrollment completion
        enrollment_reduction = features.enrollment_completion_pct * 0.05

        # Increase uncertainty if signals are inconsistent
        consistency_factor = 1.0 + (1.0 - features.signal_consistency) * 0.5

        uncertainty = (base_uncertainty - signal_reduction - enrollment_reduction) * consistency_factor
        uncertainty = max(5.0, min(25.0, uncertainty))

        low = max(1.0, probability - uncertainty)
        high = min(99.0, probability + uncertainty)

        return (round(low, 1), round(high, 1))

    def _assess_risk_level(
        self,
        probability: float,
        features: TrialFeatures
    ) -> RiskLevel:
        """Assess overall risk level of the trial."""
        # Start with probability-based risk
        if probability >= 70:
            base_risk = RiskLevel.LOW
        elif probability >= 50:
            base_risk = RiskLevel.MODERATE
        elif probability >= 30:
            base_risk = RiskLevel.HIGH
        else:
            base_risk = RiskLevel.VERY_HIGH

        # Adjust for red flags
        red_flags = 0

        if features.endpoint_changes > 0:
            red_flags += 2

        if features.completion_date_changes >= 2:
            red_flags += 1

        if features.enrollment_rate < 0.7:
            red_flags += 1

        if features.insider_sell_signals > features.insider_buy_signals + 2:
            red_flags += 1

        # Escalate risk level based on red flags
        risk_levels = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.VERY_HIGH]
        current_idx = risk_levels.index(base_risk)
        escalated_idx = min(len(risk_levels) - 1, current_idx + (red_flags // 2))

        return risk_levels[escalated_idx]

    def _identify_key_factors(
        self,
        features: TrialFeatures,
        contributions: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Identify key factors influencing prediction."""
        factors = []

        # Sort contributions by absolute value
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        for name, value in sorted_contributions[:5]:
            if abs(value) < 0.01:
                continue

            direction = "positive" if value > 0 else "negative"
            impact = "high" if abs(value) > 0.1 else "moderate" if abs(value) > 0.05 else "low"

            # Generate human-readable description
            description = self._get_factor_description(name, value, features)

            factors.append({
                "factor": name,
                "direction": direction,
                "impact": impact,
                "contribution": round(value, 3),
                "description": description,
            })

        return factors

    def _get_factor_description(
        self,
        factor_name: str,
        value: float,
        features: TrialFeatures
    ) -> str:
        """Generate human-readable description for a factor."""
        descriptions = {
            "enrollment_rate": (
                f"Enrollment is {'ahead of' if features.enrollment_rate > 1 else 'behind'} schedule "
                f"({features.enrollment_rate:.0%} of expected rate)"
            ),
            "enrollment_completion": (
                f"Enrollment is {features.enrollment_completion_pct:.0f}% complete"
            ),
            "timeline_adherence": (
                f"Trial timeline adherence is {features.timeline_adherence:.0%}"
            ),
            "completion_date_changes": (
                f"Completion date changed {features.completion_date_changes} time(s)"
            ),
            "endpoint_changes": (
                f"Primary endpoint changed {features.endpoint_changes} time(s) - critical red flag"
            ),
            "protocol_amendments": (
                f"{features.protocol_amendments} protocol amendment(s) detected"
            ),
            "site_changes": (
                f"Trial sites {'increased' if features.site_changes > 0 else 'decreased'} by {abs(features.site_changes)}"
            ),
            "insider_activity": (
                f"Net insider {'buying' if value > 0 else 'selling'} activity detected"
            ),
            "sec_signals": (
                f"SEC filings indicate {'positive' if value > 0 else 'negative'} sentiment"
            ),
            "signal_momentum": (
                f"Consistent {'positive' if value > 0 else 'negative'} signal momentum"
            ),
        }

        return descriptions.get(factor_name, f"{factor_name}: {value:.3f}")

    def _generate_warnings(
        self,
        features: TrialFeatures,
        probability: float
    ) -> List[str]:
        """Generate warnings about potential issues."""
        warnings = []

        if features.endpoint_changes > 0:
            warnings.append(
                "CRITICAL: Primary endpoint has been changed - historically a major red flag"
            )

        if features.enrollment_rate < 0.5:
            warnings.append(
                "WARNING: Enrollment is significantly behind schedule (<50% of expected rate)"
            )

        if features.completion_date_changes >= 3:
            warnings.append(
                "WARNING: Completion date has been changed multiple times"
            )

        if features.insider_sell_signals > features.insider_buy_signals + 3:
            warnings.append(
                "WARNING: Significant insider selling activity detected"
            )

        if features.total_signal_count < 3:
            warnings.append(
                "NOTE: Limited signal data available - prediction confidence is lower"
            )

        if probability < 30:
            warnings.append(
                "ALERT: Low success probability - consider risk management"
            )

        return warnings

    def predict_batch(
        self,
        trials: List[Tuple[Dict, List[Dict]]]
    ) -> List[PredictionResult]:
        """
        Predict outcomes for multiple trials.

        Args:
            trials: List of (trial_data, signals) tuples

        Returns:
            List of PredictionResult objects
        """
        results = []

        for trial_data, signals in trials:
            try:
                result = self.predict(trial_data, signals)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict trial {trial_data.get('trial_id')}: {e}")

        return results


if __name__ == "__main__":
    # Test the predictor
    logging.basicConfig(level=logging.INFO)

    print("Testing Trial Outcome Predictor...")
    print("=" * 60)

    predictor = TrialPredictor()

    # Sample trial data - positive scenario
    positive_trial = {
        "trial_id": "NCT12345678",
        "drug_name": "TestDrug-1",
        "indication": "non-small cell lung cancer",
        "phase": "PHASE3",
        "sponsor": "Pfizer Inc",
        "enrollment_target": 500,
        "enrollment_current": 480,
        "start_date": date(2023, 1, 1),
        "expected_completion": date(2025, 6, 30),
        "primary_completion_date": date(2025, 3, 31),
    }

    positive_signals = [
        {"signal_type": "sites_added", "signal_weight": 3},
        {"signal_type": "early_enrollment", "signal_weight": 3},
        {"signal_type": "insider_buying", "signal_weight": 4},
        {"signal_type": "sec_8k_positive", "signal_weight": 3},
    ]

    print("\nPositive Scenario Trial:")
    print(f"  Drug: {positive_trial['drug_name']}")
    print(f"  Indication: {positive_trial['indication']}")
    print(f"  Enrollment: {positive_trial['enrollment_current']}/{positive_trial['enrollment_target']}")

    result = predictor.predict(positive_trial, positive_signals)

    print(f"\nPrediction Results:")
    print(f"  Success Probability: {result.success_probability}%")
    print(f"  Confidence Interval: {result.confidence_interval[0]}% - {result.confidence_interval[1]}%")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"\n  Key Factors:")
    for factor in result.key_factors[:3]:
        print(f"    - {factor['description']} ({factor['direction']}, {factor['impact']} impact)")
    if result.warnings:
        print(f"\n  Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")

    # Sample trial data - negative scenario
    print("\n" + "=" * 60)
    print("\nNegative Scenario Trial:")

    negative_trial = {
        "trial_id": "NCT87654321",
        "drug_name": "TestDrug-2",
        "indication": "Alzheimer's disease",
        "phase": "PHASE3",
        "sponsor": "Small Biotech Inc",
        "enrollment_target": 800,
        "enrollment_current": 300,
        "start_date": date(2022, 1, 1),
        "expected_completion": date(2024, 12, 31),
        "primary_completion_date": date(2024, 6, 30),
    }

    negative_signals = [
        {"signal_type": "endpoint_change", "signal_weight": -5},
        {"signal_type": "enrollment_extended", "signal_weight": -3},
        {"signal_type": "insider_selling", "signal_weight": -4},
        {"signal_type": "insider_selling", "signal_weight": -4},
        {"signal_type": "completion_date_delayed", "signal_weight": -3},
    ]

    print(f"  Drug: {negative_trial['drug_name']}")
    print(f"  Indication: {negative_trial['indication']}")
    print(f"  Enrollment: {negative_trial['enrollment_current']}/{negative_trial['enrollment_target']}")

    result = predictor.predict(negative_trial, negative_signals)

    print(f"\nPrediction Results:")
    print(f"  Success Probability: {result.success_probability}%")
    print(f"  Confidence Interval: {result.confidence_interval[0]}% - {result.confidence_interval[1]}%")
    print(f"  Risk Level: {result.risk_level.value}")
    print(f"\n  Key Factors:")
    for factor in result.key_factors[:3]:
        print(f"    - {factor['description']} ({factor['direction']}, {factor['impact']} impact)")
    if result.warnings:
        print(f"\n  Warnings:")
        for warning in result.warnings:
            print(f"    - {warning}")

    print("\n" + "=" * 60)
    print("Test complete!")
