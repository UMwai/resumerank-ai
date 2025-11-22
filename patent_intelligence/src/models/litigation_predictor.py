"""
Litigation Outcome Prediction Model

Predicts probability of patent being upheld vs invalidated in Hatch-Waxman
litigation using historical case data and patent characteristics.

Features:
- Patent age at litigation
- Number of claims
- Court jurisdiction
- Challenger history (success rate)
- Patent type (composition vs method)
- Therapeutic area
- Revenue at stake
- Prior art density
"""

import json
import pickle
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import sklearn, handle gracefully if not available
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
        confusion_matrix,
    )
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Install with: pip install scikit-learn")


@dataclass
class LitigationFeatures:
    """Features for litigation outcome prediction."""

    # Patent characteristics
    patent_age_at_litigation: int  # Years since patent grant
    claim_count: int  # Total number of claims
    independent_claims: int
    patent_type: str  # COMPOSITION, METHOD_OF_USE, FORMULATION

    # Case characteristics
    court_jurisdiction: str  # District court (e.g., D. Del., N.D. Cal.)
    challenger_company: str
    challenger_win_rate: float  # Historical win rate (0-1)

    # Market factors
    annual_revenue: int  # Drug revenue at stake
    therapeutic_area: str
    num_prior_challenges: int  # Previous challenges to this patent

    # Technical factors
    prior_art_density: float  # 0-1, estimate of relevant prior art
    claim_breadth: str  # narrow, moderate, broad

    # Optional fields
    has_pte: bool = False  # Patent term extension
    has_pediatric_exclusivity: bool = False
    is_biologic: bool = False


@dataclass
class LitigationPrediction:
    """Prediction result for litigation outcome."""

    patent_number: str
    drug_name: str

    # Predictions
    upheld_probability: float  # 0-100
    invalidated_probability: float  # 0-100
    predicted_outcome: str  # UPHELD, INVALIDATED

    # Confidence and uncertainty
    confidence: float  # 0-100
    prediction_interval_low: float  # Lower bound of prediction
    prediction_interval_high: float  # Upper bound of prediction

    # Feature importance
    top_factors: List[Tuple[str, float]]  # (feature_name, importance)
    risk_factors: List[str]  # Key risk factors identified

    # Model info
    model_type: str
    model_accuracy: float
    training_samples: int

    # Comparison to similar cases
    similar_cases: List[Dict[str, Any]]

    # Timestamp
    prediction_date: date = field(default_factory=date.today)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["prediction_date"] = self.prediction_date.isoformat()
        return result


# Historical Hatch-Waxman litigation data for training
HISTORICAL_LITIGATION_DATA = [
    # 2018-2024 historical cases with outcomes
    {
        "patent_number": "6197819",
        "drug_name": "Lyrica",
        "outcome": "INVALIDATED",
        "patent_age_at_litigation": 15,
        "claim_count": 22,
        "independent_claims": 4,
        "patent_type": "METHOD_OF_USE",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Teva",
        "challenger_win_rate": 0.55,
        "annual_revenue": 5070000000,
        "therapeutic_area": "Neurology",
        "num_prior_challenges": 2,
        "prior_art_density": 0.7,
        "claim_breadth": "broad",
        "has_pte": False,
        "has_pediatric_exclusivity": True,
        "is_biologic": False,
    },
    {
        "patent_number": "8399514",
        "drug_name": "Tecfidera",
        "outcome": "INVALIDATED",
        "patent_age_at_litigation": 6,
        "claim_count": 28,
        "independent_claims": 5,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "N.D. W.Va.",
        "challenger_company": "Mylan",
        "challenger_win_rate": 0.48,
        "annual_revenue": 4400000000,
        "therapeutic_area": "Neurology",
        "num_prior_challenges": 1,
        "prior_art_density": 0.65,
        "claim_breadth": "moderate",
        "has_pte": False,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "6090382",
        "drug_name": "Humira",
        "outcome": "SETTLED",
        "patent_age_at_litigation": 18,
        "claim_count": 33,
        "independent_claims": 6,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Amgen",
        "challenger_win_rate": 0.42,
        "annual_revenue": 21237000000,
        "therapeutic_area": "Immunology",
        "num_prior_challenges": 5,
        "prior_art_density": 0.5,
        "claim_breadth": "moderate",
        "has_pte": True,
        "has_pediatric_exclusivity": True,
        "is_biologic": True,
    },
    {
        "patent_number": "5635517",
        "drug_name": "Revlimid",
        "outcome": "SETTLED",
        "patent_age_at_litigation": 20,
        "claim_count": 18,
        "independent_claims": 3,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. N.J.",
        "challenger_company": "Teva",
        "challenger_win_rate": 0.55,
        "annual_revenue": 12100000000,
        "therapeutic_area": "Oncology",
        "num_prior_challenges": 3,
        "prior_art_density": 0.4,
        "claim_breadth": "narrow",
        "has_pte": False,
        "has_pediatric_exclusivity": True,
        "is_biologic": False,
    },
    {
        "patent_number": "7371746",
        "drug_name": "Eliquis",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 10,
        "claim_count": 42,
        "independent_claims": 8,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Unichem",
        "challenger_win_rate": 0.35,
        "annual_revenue": 12200000000,
        "therapeutic_area": "Cardiovascular",
        "num_prior_challenges": 4,
        "prior_art_density": 0.35,
        "claim_breadth": "narrow",
        "has_pte": True,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "7585860",
        "drug_name": "Xarelto",
        "outcome": "SETTLED",
        "patent_age_at_litigation": 12,
        "claim_count": 35,
        "independent_claims": 6,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Teva",
        "challenger_win_rate": 0.55,
        "annual_revenue": 6400000000,
        "therapeutic_area": "Cardiovascular",
        "num_prior_challenges": 3,
        "prior_art_density": 0.45,
        "claim_breadth": "moderate",
        "has_pte": True,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "7863278",
        "drug_name": "Ibrance",
        "outcome": "SETTLED",
        "patent_age_at_litigation": 10,
        "claim_count": 25,
        "independent_claims": 5,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Teva",
        "challenger_win_rate": 0.55,
        "annual_revenue": 5400000000,
        "therapeutic_area": "Oncology",
        "num_prior_challenges": 2,
        "prior_art_density": 0.5,
        "claim_breadth": "moderate",
        "has_pte": False,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "7303746",
        "drug_name": "Eylea",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 12,
        "claim_count": 38,
        "independent_claims": 7,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Samsung Bioepis",
        "challenger_win_rate": 0.30,
        "annual_revenue": 9900000000,
        "therapeutic_area": "Ophthalmology",
        "num_prior_challenges": 2,
        "prior_art_density": 0.3,
        "claim_breadth": "narrow",
        "has_pte": True,
        "has_pediatric_exclusivity": False,
        "is_biologic": True,
    },
    {
        "patent_number": "6902734",
        "drug_name": "Stelara",
        "outcome": "SETTLED",
        "patent_age_at_litigation": 16,
        "claim_count": 30,
        "independent_claims": 5,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. N.J.",
        "challenger_company": "Amgen",
        "challenger_win_rate": 0.42,
        "annual_revenue": 10400000000,
        "therapeutic_area": "Immunology",
        "num_prior_challenges": 4,
        "prior_art_density": 0.45,
        "claim_breadth": "moderate",
        "has_pte": True,
        "has_pediatric_exclusivity": True,
        "is_biologic": True,
    },
    {
        "patent_number": "7709517",
        "drug_name": "Xtandi",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 8,
        "claim_count": 45,
        "independent_claims": 9,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Hikma",
        "challenger_win_rate": 0.38,
        "annual_revenue": 4900000000,
        "therapeutic_area": "Oncology",
        "num_prior_challenges": 1,
        "prior_art_density": 0.25,
        "claim_breadth": "narrow",
        "has_pte": True,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "7514444",
        "drug_name": "Imbruvica",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 7,
        "claim_count": 52,
        "independent_claims": 10,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Dr. Reddys",
        "challenger_win_rate": 0.40,
        "annual_revenue": 5400000000,
        "therapeutic_area": "Oncology",
        "num_prior_challenges": 2,
        "prior_art_density": 0.3,
        "claim_breadth": "narrow",
        "has_pte": False,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "8101659",
        "drug_name": "Entresto",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 9,
        "claim_count": 48,
        "independent_claims": 8,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. N.J.",
        "challenger_company": "Zydus",
        "challenger_win_rate": 0.35,
        "annual_revenue": 5600000000,
        "therapeutic_area": "Cardiovascular",
        "num_prior_challenges": 1,
        "prior_art_density": 0.35,
        "claim_breadth": "narrow",
        "has_pte": True,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    # Add more synthetic cases for better training
    {
        "patent_number": "SYNTH001",
        "drug_name": "SynthDrug1",
        "outcome": "INVALIDATED",
        "patent_age_at_litigation": 18,
        "claim_count": 15,
        "independent_claims": 2,
        "patent_type": "METHOD_OF_USE",
        "court_jurisdiction": "E.D. Tex.",
        "challenger_company": "Apotex",
        "challenger_win_rate": 0.52,
        "annual_revenue": 2000000000,
        "therapeutic_area": "Cardiology",
        "num_prior_challenges": 4,
        "prior_art_density": 0.75,
        "claim_breadth": "broad",
        "has_pte": False,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "SYNTH002",
        "drug_name": "SynthDrug2",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 5,
        "claim_count": 55,
        "independent_claims": 12,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Sandoz",
        "challenger_win_rate": 0.38,
        "annual_revenue": 3500000000,
        "therapeutic_area": "Oncology",
        "num_prior_challenges": 1,
        "prior_art_density": 0.2,
        "claim_breadth": "narrow",
        "has_pte": True,
        "has_pediatric_exclusivity": True,
        "is_biologic": False,
    },
    {
        "patent_number": "SYNTH003",
        "drug_name": "SynthDrug3",
        "outcome": "INVALIDATED",
        "patent_age_at_litigation": 16,
        "claim_count": 20,
        "independent_claims": 3,
        "patent_type": "FORMULATION",
        "court_jurisdiction": "N.D. Cal.",
        "challenger_company": "Par Pharma",
        "challenger_win_rate": 0.45,
        "annual_revenue": 1500000000,
        "therapeutic_area": "Neurology",
        "num_prior_challenges": 3,
        "prior_art_density": 0.6,
        "claim_breadth": "moderate",
        "has_pte": False,
        "has_pediatric_exclusivity": False,
        "is_biologic": False,
    },
    {
        "patent_number": "SYNTH004",
        "drug_name": "SynthDrug4",
        "outcome": "UPHELD",
        "patent_age_at_litigation": 6,
        "claim_count": 60,
        "independent_claims": 15,
        "patent_type": "COMPOSITION",
        "court_jurisdiction": "D. Del.",
        "challenger_company": "Lupin",
        "challenger_win_rate": 0.42,
        "annual_revenue": 8000000000,
        "therapeutic_area": "Immunology",
        "num_prior_challenges": 0,
        "prior_art_density": 0.15,
        "claim_breadth": "narrow",
        "has_pte": True,
        "has_pediatric_exclusivity": True,
        "is_biologic": True,
    },
]


class LitigationPredictor:
    """
    ML model to predict Hatch-Waxman litigation outcomes.

    Uses historical case data to predict whether a patent will be
    upheld or invalidated based on patent and case characteristics.
    """

    # Court jurisdiction mapping for encoding
    COURT_CODES = {
        "D. Del.": 0,  # District of Delaware (most patent cases)
        "D. N.J.": 1,  # District of New Jersey
        "N.D. Cal.": 2,  # Northern District of California
        "E.D. Tex.": 3,  # Eastern District of Texas
        "N.D. W.Va.": 4,  # Northern District of West Virginia
        "OTHER": 5,
    }

    # Patent type mapping
    PATENT_TYPE_CODES = {
        "COMPOSITION": 0,
        "METHOD_OF_USE": 1,
        "FORMULATION": 2,
        "OTHER": 3,
    }

    # Therapeutic area mapping
    THERAPEUTIC_CODES = {
        "Oncology": 0,
        "Immunology": 1,
        "Cardiovascular": 2,
        "Neurology": 3,
        "Ophthalmology": 4,
        "Diabetes": 5,
        "OTHER": 6,
    }

    # Claim breadth mapping
    BREADTH_CODES = {
        "narrow": 0,
        "moderate": 1,
        "broad": 2,
    }

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the predictor.

        Args:
            model_path: Path to saved model. If None, trains new model.
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            "patent_age_at_litigation",
            "claim_count",
            "independent_claims",
            "patent_type_encoded",
            "court_encoded",
            "challenger_win_rate",
            "annual_revenue_scaled",
            "therapeutic_encoded",
            "num_prior_challenges",
            "prior_art_density",
            "claim_breadth_encoded",
            "has_pte",
            "has_pediatric_exclusivity",
            "is_biologic",
        ]
        self.model_accuracy = 0.0
        self.training_samples = 0

        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._train_model()

    def _encode_features(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Encode categorical features and create feature vector.

        Args:
            data: Raw feature dictionary.

        Returns:
            Numpy array of encoded features.
        """
        # Encode court
        court = data.get("court_jurisdiction", "OTHER")
        court_encoded = self.COURT_CODES.get(court, self.COURT_CODES["OTHER"])

        # Encode patent type
        patent_type = data.get("patent_type", "OTHER")
        type_encoded = self.PATENT_TYPE_CODES.get(patent_type, self.PATENT_TYPE_CODES["OTHER"])

        # Encode therapeutic area
        therapeutic = data.get("therapeutic_area", "OTHER")
        therapeutic_encoded = self.THERAPEUTIC_CODES.get(
            therapeutic, self.THERAPEUTIC_CODES["OTHER"]
        )

        # Encode claim breadth
        breadth = data.get("claim_breadth", "moderate")
        breadth_encoded = self.BREADTH_CODES.get(breadth, 1)

        # Scale revenue (log transform for large values)
        revenue = data.get("annual_revenue", 1000000000)
        revenue_scaled = np.log10(max(revenue, 1))

        features = np.array([
            data.get("patent_age_at_litigation", 10),
            data.get("claim_count", 20),
            data.get("independent_claims", 5),
            type_encoded,
            court_encoded,
            data.get("challenger_win_rate", 0.5),
            revenue_scaled,
            therapeutic_encoded,
            data.get("num_prior_challenges", 1),
            data.get("prior_art_density", 0.5),
            breadth_encoded,
            int(data.get("has_pte", False)),
            int(data.get("has_pediatric_exclusivity", False)),
            int(data.get("is_biologic", False)),
        ])

        return features

    def _train_model(self) -> None:
        """Train the prediction model on historical data."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using rule-based prediction")
            return

        logger.info("Training litigation prediction model...")

        # Prepare training data
        X = []
        y = []

        for case in HISTORICAL_LITIGATION_DATA:
            features = self._encode_features(case)
            X.append(features)

            # Convert outcome to binary: 1 = UPHELD, 0 = INVALIDATED/SETTLED
            # SETTLED cases are treated as partial wins for challenger
            outcome = case["outcome"]
            if outcome == "UPHELD":
                y.append(1)
            else:  # INVALIDATED or SETTLED
                y.append(0)

        X = np.array(X)
        y = np.array(y)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        logger.info(f"Cross-validation scores: {cv_scores}")
        logger.info(f"Mean CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

        # Fit on full data
        self.model.fit(X_scaled, y)

        # Store metrics
        self.model_accuracy = cv_scores.mean() * 100
        self.training_samples = len(X)

        logger.info(
            f"Model trained on {self.training_samples} cases, "
            f"accuracy: {self.model_accuracy:.1f}%"
        )

    def predict(
        self,
        patent_number: str,
        drug_name: str,
        features: LitigationFeatures,
    ) -> LitigationPrediction:
        """
        Predict litigation outcome for a patent.

        Args:
            patent_number: Patent number.
            drug_name: Drug name.
            features: LitigationFeatures with case characteristics.

        Returns:
            LitigationPrediction with probabilities and analysis.
        """
        # Convert features to dict
        feature_dict = {
            "patent_age_at_litigation": features.patent_age_at_litigation,
            "claim_count": features.claim_count,
            "independent_claims": features.independent_claims,
            "patent_type": features.patent_type,
            "court_jurisdiction": features.court_jurisdiction,
            "challenger_win_rate": features.challenger_win_rate,
            "annual_revenue": features.annual_revenue,
            "therapeutic_area": features.therapeutic_area,
            "num_prior_challenges": features.num_prior_challenges,
            "prior_art_density": features.prior_art_density,
            "claim_breadth": features.claim_breadth,
            "has_pte": features.has_pte,
            "has_pediatric_exclusivity": features.has_pediatric_exclusivity,
            "is_biologic": features.is_biologic,
        }

        # Use ML model if available
        if self.model is not None and SKLEARN_AVAILABLE:
            X = self._encode_features(feature_dict).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

            # Get probability
            proba = self.model.predict_proba(X_scaled)[0]
            upheld_prob = proba[1] * 100 if len(proba) > 1 else 50.0
            invalidated_prob = 100 - upheld_prob

            # Get feature importance
            importances = self.model.feature_importances_
            top_factors = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )[:5]

        else:
            # Rule-based fallback
            upheld_prob, top_factors = self._rule_based_prediction(feature_dict)
            invalidated_prob = 100 - upheld_prob

        # Determine predicted outcome
        predicted_outcome = "UPHELD" if upheld_prob > 50 else "INVALIDATED"

        # Calculate confidence based on how far from 50%
        confidence = abs(upheld_prob - 50) * 2

        # Calculate prediction interval (simple heuristic)
        interval_width = 30 - (confidence / 100 * 20)
        prediction_interval_low = max(0, upheld_prob - interval_width)
        prediction_interval_high = min(100, upheld_prob + interval_width)

        # Identify risk factors
        risk_factors = self._identify_risk_factors(feature_dict)

        # Find similar cases
        similar_cases = self._find_similar_cases(feature_dict)

        return LitigationPrediction(
            patent_number=patent_number,
            drug_name=drug_name,
            upheld_probability=round(upheld_prob, 1),
            invalidated_probability=round(invalidated_prob, 1),
            predicted_outcome=predicted_outcome,
            confidence=round(confidence, 1),
            prediction_interval_low=round(prediction_interval_low, 1),
            prediction_interval_high=round(prediction_interval_high, 1),
            top_factors=[(name, round(imp * 100, 1)) for name, imp in top_factors],
            risk_factors=risk_factors,
            model_type="GradientBoosting" if self.model else "RuleBased",
            model_accuracy=round(self.model_accuracy, 1),
            training_samples=self.training_samples,
            similar_cases=similar_cases,
        )

    def _rule_based_prediction(
        self,
        features: Dict[str, Any],
    ) -> Tuple[float, List[Tuple[str, float]]]:
        """
        Rule-based prediction fallback when sklearn not available.

        Args:
            features: Feature dictionary.

        Returns:
            Tuple of (upheld_probability, top_factors).
        """
        base_prob = 50.0

        # Adjustments based on features
        adjustments = []

        # Claim characteristics (more claims = stronger)
        if features["claim_count"] > 40:
            adjustments.append(("claim_count", 10))
        elif features["claim_count"] < 20:
            adjustments.append(("claim_count", -10))

        # Patent age (older = weaker)
        if features["patent_age_at_litigation"] > 15:
            adjustments.append(("patent_age", -15))
        elif features["patent_age_at_litigation"] < 8:
            adjustments.append(("patent_age", 10))

        # Prior art density (high = weaker)
        if features["prior_art_density"] > 0.6:
            adjustments.append(("prior_art", -20))
        elif features["prior_art_density"] < 0.3:
            adjustments.append(("prior_art", 15))

        # Patent type (composition strongest)
        if features["patent_type"] == "COMPOSITION":
            adjustments.append(("patent_type", 10))
        elif features["patent_type"] == "METHOD_OF_USE":
            adjustments.append(("patent_type", -15))

        # Challenger win rate
        if features["challenger_win_rate"] > 0.5:
            adjustments.append(("challenger_strength", -10))
        elif features["challenger_win_rate"] < 0.35:
            adjustments.append(("challenger_strength", 10))

        # Extensions
        if features.get("has_pte"):
            adjustments.append(("has_pte", 5))
        if features.get("is_biologic"):
            adjustments.append(("is_biologic", 5))

        # Claim breadth
        if features["claim_breadth"] == "narrow":
            adjustments.append(("claim_breadth", 15))
        elif features["claim_breadth"] == "broad":
            adjustments.append(("claim_breadth", -10))

        # Apply adjustments
        for _, adj in adjustments:
            base_prob += adj

        # Clamp to valid range
        upheld_prob = max(10, min(90, base_prob))

        # Create importance-like scores
        top_factors = [
            (name, abs(adj) / 100)
            for name, adj in sorted(adjustments, key=lambda x: abs(x[1]), reverse=True)
        ][:5]

        return upheld_prob, top_factors

    def _identify_risk_factors(self, features: Dict[str, Any]) -> List[str]:
        """Identify key risk factors that may affect outcome."""
        risks = []

        if features["patent_age_at_litigation"] > 15:
            risks.append("Old patent - more prior art available")

        if features["prior_art_density"] > 0.5:
            risks.append("High prior art density in therapeutic area")

        if features["patent_type"] == "METHOD_OF_USE":
            risks.append("Method of use patents harder to defend than composition")

        if features["challenger_win_rate"] > 0.5:
            risks.append("Experienced challenger with strong track record")

        if features["claim_breadth"] == "broad":
            risks.append("Broad claims more vulnerable to invalidity challenge")

        if features["num_prior_challenges"] > 2:
            risks.append("Multiple prior challenges indicate vulnerability")

        return risks

    def _find_similar_cases(
        self,
        features: Dict[str, Any],
        top_n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find most similar historical cases."""
        similarities = []

        for case in HISTORICAL_LITIGATION_DATA:
            # Skip synthetic cases
            if case["patent_number"].startswith("SYNTH"):
                continue

            # Calculate similarity score
            score = 0

            # Same therapeutic area
            if case["therapeutic_area"] == features["therapeutic_area"]:
                score += 3

            # Same patent type
            if case["patent_type"] == features["patent_type"]:
                score += 2

            # Similar patent age
            age_diff = abs(case["patent_age_at_litigation"] - features["patent_age_at_litigation"])
            if age_diff < 3:
                score += 2
            elif age_diff < 5:
                score += 1

            # Similar prior art density
            density_diff = abs(case["prior_art_density"] - features["prior_art_density"])
            if density_diff < 0.15:
                score += 2
            elif density_diff < 0.3:
                score += 1

            similarities.append({
                "drug": case["drug_name"],
                "patent": case["patent_number"],
                "outcome": case["outcome"],
                "therapeutic_area": case["therapeutic_area"],
                "similarity_score": score,
            })

        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
        return similarities[:top_n]

    def save_model(self, path: str) -> None:
        """Save trained model to file."""
        if self.model is None:
            logger.warning("No model to save")
            return

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "accuracy": self.model_accuracy,
            "training_samples": self.training_samples,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load trained model from file."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.model_accuracy = model_data["accuracy"]
        self.training_samples = model_data["training_samples"]

        logger.info(f"Model loaded from {path}")

    def get_model_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        return {
            "model_type": "GradientBoosting" if self.model else "RuleBased",
            "accuracy": self.model_accuracy,
            "training_samples": self.training_samples,
            "feature_count": len(self.feature_names),
            "sklearn_available": SKLEARN_AVAILABLE,
        }


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Litigation Outcome Prediction Model")
    print("=" * 70)

    # Initialize predictor
    predictor = LitigationPredictor()

    # Show model metrics
    metrics = predictor.get_model_metrics()
    print(f"\nModel Type: {metrics['model_type']}")
    print(f"Training Samples: {metrics['training_samples']}")
    print(f"Model Accuracy: {metrics['accuracy']:.1f}%")

    # Test predictions
    test_cases = [
        {
            "patent_number": "TEST001",
            "drug_name": "Test Drug A",
            "features": LitigationFeatures(
                patent_age_at_litigation=12,
                claim_count=35,
                independent_claims=6,
                patent_type="COMPOSITION",
                court_jurisdiction="D. Del.",
                challenger_company="Teva",
                challenger_win_rate=0.55,
                annual_revenue=5000000000,
                therapeutic_area="Oncology",
                num_prior_challenges=2,
                prior_art_density=0.4,
                claim_breadth="moderate",
                has_pte=True,
                has_pediatric_exclusivity=False,
                is_biologic=False,
            ),
        },
        {
            "patent_number": "TEST002",
            "drug_name": "Test Drug B",
            "features": LitigationFeatures(
                patent_age_at_litigation=18,
                claim_count=18,
                independent_claims=3,
                patent_type="METHOD_OF_USE",
                court_jurisdiction="E.D. Tex.",
                challenger_company="Mylan",
                challenger_win_rate=0.48,
                annual_revenue=3000000000,
                therapeutic_area="Neurology",
                num_prior_challenges=4,
                prior_art_density=0.7,
                claim_breadth="broad",
                has_pte=False,
                has_pediatric_exclusivity=False,
                is_biologic=False,
            ),
        },
    ]

    for test in test_cases:
        prediction = predictor.predict(
            patent_number=test["patent_number"],
            drug_name=test["drug_name"],
            features=test["features"],
        )

        print(f"\n{'=' * 50}")
        print(f"Patent: {prediction.patent_number} - {prediction.drug_name}")
        print(f"{'=' * 50}")
        print(f"Predicted Outcome: {prediction.predicted_outcome}")
        print(f"Upheld Probability: {prediction.upheld_probability:.1f}%")
        print(f"Invalidated Probability: {prediction.invalidated_probability:.1f}%")
        print(f"Confidence: {prediction.confidence:.1f}%")
        print(f"Prediction Interval: [{prediction.prediction_interval_low:.1f}% - {prediction.prediction_interval_high:.1f}%]")

        print(f"\nTop Factors:")
        for factor, importance in prediction.top_factors:
            print(f"  - {factor}: {importance:.1f}%")

        print(f"\nRisk Factors:")
        for risk in prediction.risk_factors:
            print(f"  - {risk}")

        print(f"\nSimilar Cases:")
        for case in prediction.similar_cases:
            print(f"  - {case['drug']} ({case['outcome']})")
