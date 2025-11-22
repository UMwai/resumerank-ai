"""
Machine Learning Signal Scorer

Enhanced scoring model using ML to weight signals optimally.
Features:
- Trains on historical performance data
- Sector-specific adjustments (oncology, rare disease, etc.)
- Market regime detection (bull vs bear)
- Multi-timeframe scoring (1 week, 1 month, 3 months)
- Ensemble methods for robust predictions
"""

import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import get_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


# Try to import sklearn, but provide fallback
try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. ML features limited.")
    SKLEARN_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"          # Strong uptrend, VIX < 15
    NEUTRAL = "neutral"    # Sideways, VIX 15-25
    BEAR = "bear"          # Downtrend, VIX 25-35
    CRISIS = "crisis"      # Panic, VIX > 35


class BiotechSector(Enum):
    """Biotech sector classifications."""
    ONCOLOGY = "oncology"
    RARE_DISEASE = "rare_disease"
    CNS = "cns"
    IMMUNOLOGY = "immunology"
    GENE_THERAPY = "gene_therapy"
    INFECTIOUS = "infectious"
    CARDIOVASCULAR = "cardiovascular"
    OTHER = "other"


@dataclass
class SignalFeatures:
    """Feature vector for ML model."""
    # Insider signals
    insider_buy_count: int = 0
    insider_sell_count: int = 0
    insider_buy_value: float = 0.0
    insider_sell_value: float = 0.0
    ceo_transaction: int = 0  # 1 for buy, -1 for sell, 0 none
    cfo_transaction: int = 0
    multiple_insider_buy: bool = False
    multiple_insider_sell: bool = False
    avg_insider_signal_age: float = 0.0  # Days

    # Institutional signals
    top_fund_new_positions: int = 0
    top_fund_increases: int = 0
    top_fund_decreases: int = 0
    top_fund_exits: int = 0
    institutional_convergence: bool = False

    # Hiring signals
    total_jobs_30d: int = 0
    commercial_jobs: int = 0
    clinical_jobs: int = 0
    rd_jobs: int = 0
    senior_hires: int = 0
    hiring_velocity: float = 0.0  # Jobs per week

    # Sentiment signals
    glassdoor_sentiment: float = 0.0
    sentiment_trend: float = 0.0  # Change over time
    layoff_mentions: int = 0

    # Executive signals
    executive_departures: int = 0
    executive_hires: int = 0

    # Market context
    market_regime: str = "neutral"
    sector: str = "other"
    days_since_last_catalyst: int = 365
    upcoming_catalyst_days: int = 365

    # Technical (if available)
    price_momentum_30d: float = 0.0
    volume_ratio: float = 1.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model."""
        # Map categorical to numeric
        regime_map = {'bull': 1, 'neutral': 0, 'bear': -1, 'crisis': -2}
        sector_map = {
            'oncology': 1, 'rare_disease': 2, 'cns': 3, 'immunology': 4,
            'gene_therapy': 5, 'infectious': 6, 'cardiovascular': 7, 'other': 0
        }

        return np.array([
            self.insider_buy_count,
            self.insider_sell_count,
            np.log1p(self.insider_buy_value),
            np.log1p(self.insider_sell_value),
            self.ceo_transaction,
            self.cfo_transaction,
            int(self.multiple_insider_buy),
            int(self.multiple_insider_sell),
            self.avg_insider_signal_age,
            self.top_fund_new_positions,
            self.top_fund_increases,
            self.top_fund_decreases,
            self.top_fund_exits,
            int(self.institutional_convergence),
            self.total_jobs_30d,
            self.commercial_jobs,
            self.clinical_jobs,
            self.rd_jobs,
            self.senior_hires,
            self.hiring_velocity,
            self.glassdoor_sentiment,
            self.sentiment_trend,
            self.layoff_mentions,
            self.executive_departures,
            self.executive_hires,
            regime_map.get(self.market_regime, 0),
            sector_map.get(self.sector, 0),
            np.log1p(self.days_since_last_catalyst),
            np.log1p(self.upcoming_catalyst_days),
            self.price_momentum_30d,
            self.volume_ratio,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Return feature names for interpretability."""
        return [
            'insider_buy_count', 'insider_sell_count',
            'log_insider_buy_value', 'log_insider_sell_value',
            'ceo_transaction', 'cfo_transaction',
            'multiple_insider_buy', 'multiple_insider_sell',
            'avg_insider_signal_age',
            'top_fund_new_positions', 'top_fund_increases',
            'top_fund_decreases', 'top_fund_exits',
            'institutional_convergence',
            'total_jobs_30d', 'commercial_jobs', 'clinical_jobs',
            'rd_jobs', 'senior_hires', 'hiring_velocity',
            'glassdoor_sentiment', 'sentiment_trend', 'layoff_mentions',
            'executive_departures', 'executive_hires',
            'market_regime', 'sector',
            'log_days_since_catalyst', 'log_upcoming_catalyst',
            'price_momentum_30d', 'volume_ratio',
        ]


@dataclass
class MLScore:
    """ML model prediction result."""
    ticker: str
    score_date: date
    ml_score: float  # -10 to +10
    confidence: float  # 0 to 1
    prediction_7d: float  # Expected return 7 days
    prediction_30d: float  # Expected return 30 days
    prediction_90d: float  # Expected return 90 days
    feature_contributions: Dict[str, float]
    model_agreement: float  # Agreement between ensemble models
    recommendation: str
    timeframe_scores: Dict[str, float]  # Scores for each timeframe


class MarketRegimeDetector:
    """
    Detects current market regime using VIX and market trends.

    Regimes:
    - Bull: VIX < 15, positive trend
    - Neutral: VIX 15-25, mixed signals
    - Bear: VIX 25-35, negative trend
    - Crisis: VIX > 35, extreme fear
    """

    def __init__(self):
        self._cached_regime: Optional[MarketRegime] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=4)

    def detect_regime(
        self,
        vix_level: Optional[float] = None,
        spy_return_30d: Optional[float] = None,
        xbi_return_30d: Optional[float] = None
    ) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            vix_level: Current VIX level (will fetch if not provided)
            spy_return_30d: S&P 500 30-day return
            xbi_return_30d: XBI (Biotech ETF) 30-day return

        Returns:
            MarketRegime enum
        """
        # Use cached if recent
        if self._cached_regime and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._cached_regime

        # Default values if not provided
        if vix_level is None:
            vix_level = 18  # Assume neutral

        if spy_return_30d is None:
            spy_return_30d = 0

        if xbi_return_30d is None:
            xbi_return_30d = 0

        # Determine regime
        if vix_level > 35:
            regime = MarketRegime.CRISIS
        elif vix_level > 25 or (xbi_return_30d < -15):
            regime = MarketRegime.BEAR
        elif vix_level < 15 and spy_return_30d > 3 and xbi_return_30d > 0:
            regime = MarketRegime.BULL
        else:
            regime = MarketRegime.NEUTRAL

        self._cached_regime = regime
        self._cache_time = datetime.now()

        return regime

    def get_regime_adjustments(self, regime: MarketRegime) -> Dict[str, float]:
        """
        Get signal weight adjustments for current regime.

        In bear markets: bearish signals weighted more heavily
        In bull markets: bullish signals weighted more heavily
        """
        adjustments = {
            MarketRegime.BULL: {
                'bullish_multiplier': 1.2,
                'bearish_multiplier': 0.8,
                'confidence_adjustment': 0.1,
            },
            MarketRegime.NEUTRAL: {
                'bullish_multiplier': 1.0,
                'bearish_multiplier': 1.0,
                'confidence_adjustment': 0.0,
            },
            MarketRegime.BEAR: {
                'bullish_multiplier': 0.8,
                'bearish_multiplier': 1.3,
                'confidence_adjustment': -0.1,
            },
            MarketRegime.CRISIS: {
                'bullish_multiplier': 0.5,
                'bearish_multiplier': 1.5,
                'confidence_adjustment': -0.2,
            },
        }
        return adjustments.get(regime, adjustments[MarketRegime.NEUTRAL])


class SectorAdjuster:
    """
    Applies sector-specific signal adjustments.

    Different sectors have different signal sensitivities:
    - Oncology: Clinical hiring very important
    - Rare Disease: Commercial buildup critical for launch
    - Gene Therapy: R&D signals matter more
    """

    SECTOR_WEIGHTS = {
        BiotechSector.ONCOLOGY: {
            'clinical_jobs_weight': 1.5,
            'commercial_jobs_weight': 1.2,
            'insider_buy_weight': 1.0,
            'fund_weight': 1.0,
        },
        BiotechSector.RARE_DISEASE: {
            'clinical_jobs_weight': 1.0,
            'commercial_jobs_weight': 1.8,  # Very important for launch
            'insider_buy_weight': 1.2,
            'fund_weight': 1.1,
        },
        BiotechSector.GENE_THERAPY: {
            'clinical_jobs_weight': 1.2,
            'commercial_jobs_weight': 0.8,  # Usually early stage
            'insider_buy_weight': 1.3,      # Insiders know pipeline
            'fund_weight': 1.2,
        },
        BiotechSector.CNS: {
            'clinical_jobs_weight': 1.3,
            'commercial_jobs_weight': 1.0,
            'insider_buy_weight': 1.1,
            'fund_weight': 1.0,
        },
        BiotechSector.IMMUNOLOGY: {
            'clinical_jobs_weight': 1.1,
            'commercial_jobs_weight': 1.3,
            'insider_buy_weight': 1.0,
            'fund_weight': 1.0,
        },
    }

    # Ticker to sector mapping
    TICKER_SECTORS = {
        'MRNA': BiotechSector.INFECTIOUS,
        'VRTX': BiotechSector.RARE_DISEASE,
        'CRSP': BiotechSector.GENE_THERAPY,
        'EDIT': BiotechSector.GENE_THERAPY,
        'NTLA': BiotechSector.GENE_THERAPY,
        'BEAM': BiotechSector.GENE_THERAPY,
        'REGN': BiotechSector.IMMUNOLOGY,
        'BIIB': BiotechSector.CNS,
        'ALNY': BiotechSector.RARE_DISEASE,
        'BMRN': BiotechSector.RARE_DISEASE,
        'NBIX': BiotechSector.CNS,
        'SGEN': BiotechSector.ONCOLOGY,
        'INCY': BiotechSector.ONCOLOGY,
        'SAGE': BiotechSector.CNS,
        'AXSM': BiotechSector.CNS,
    }

    def get_sector(self, ticker: str) -> BiotechSector:
        """Get sector for a ticker."""
        return self.TICKER_SECTORS.get(ticker.upper(), BiotechSector.OTHER)

    def get_sector_weights(self, sector: BiotechSector) -> Dict[str, float]:
        """Get signal weights for a sector."""
        return self.SECTOR_WEIGHTS.get(
            sector,
            {
                'clinical_jobs_weight': 1.0,
                'commercial_jobs_weight': 1.0,
                'insider_buy_weight': 1.0,
                'fund_weight': 1.0,
            }
        )


class MLSignalScorer:
    """
    Machine Learning-enhanced signal scorer.

    Uses ensemble of models:
    - Gradient Boosting for non-linear patterns
    - Random Forest for robustness
    - Ridge Regression for interpretability
    - ElasticNet for feature selection

    Features multi-timeframe predictions and sector adjustments.
    """

    TIMEFRAMES = {
        '1w': 7,
        '1m': 30,
        '3m': 90,
    }

    def __init__(self, config_path: Optional[str] = None):
        """Initialize ML scorer."""
        self.config = get_config(config_path) if config_path else None
        self.regime_detector = MarketRegimeDetector()
        self.sector_adjuster = SectorAdjuster()

        # Initialize models (if sklearn available)
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self._is_trained = False

        if SKLEARN_AVAILABLE:
            self._initialize_models()

    def _initialize_models(self):
        """Initialize ensemble models for each timeframe."""
        for timeframe in self.TIMEFRAMES:
            self.models[timeframe] = {
                'gbr': GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=4,
                    learning_rate=0.1,
                    random_state=42
                ),
                'rf': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=6,
                    random_state=42
                ),
                'ridge': Ridge(alpha=1.0),
                'elastic': ElasticNet(alpha=0.5, l1_ratio=0.5, random_state=42),
            }
            self.scalers[timeframe] = StandardScaler()

    def train(
        self,
        training_data: List[Tuple[SignalFeatures, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Train models on historical data.

        Args:
            training_data: List of (features, returns) tuples
                returns is dict with keys '1w', '1m', '3m'

        Returns:
            Dict with training metrics
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using rule-based scoring")
            return {'status': 'sklearn_unavailable'}

        if len(training_data) < 50:
            logger.warning(f"Insufficient training data: {len(training_data)} samples")
            return {'status': 'insufficient_data', 'samples': len(training_data)}

        # Prepare data
        X = np.array([f.to_array() for f, _ in training_data])
        y_dict = {tf: [] for tf in self.TIMEFRAMES}

        for _, returns in training_data:
            for tf in self.TIMEFRAMES:
                y_dict[tf].append(returns.get(tf, 0))

        metrics = {}

        # Train models for each timeframe
        for timeframe in self.TIMEFRAMES:
            y = np.array(y_dict[timeframe])

            # Scale features
            X_scaled = self.scalers[timeframe].fit_transform(X)

            # Train each model in ensemble
            timeframe_metrics = {}

            for model_name, model in self.models[timeframe].items():
                try:
                    # Time series split for validation
                    tscv = TimeSeriesSplit(n_splits=5)
                    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                               scoring='neg_mean_squared_error')

                    # Train on full data
                    model.fit(X_scaled, y)

                    # Store metrics
                    timeframe_metrics[f'{model_name}_cv_rmse'] = np.sqrt(-cv_scores.mean())
                    timeframe_metrics[f'{model_name}_cv_std'] = cv_scores.std()

                except Exception as e:
                    logger.error(f"Failed to train {model_name} for {timeframe}: {e}")

            metrics[timeframe] = timeframe_metrics

        self._is_trained = True
        logger.info(f"Trained models on {len(training_data)} samples")

        return metrics

    def score(
        self,
        ticker: str,
        features: SignalFeatures,
        market_data: Optional[Dict] = None
    ) -> MLScore:
        """
        Generate ML-enhanced score for a company.

        Args:
            ticker: Company ticker
            features: Signal features
            market_data: Optional market context (VIX, etc.)

        Returns:
            MLScore with predictions
        """
        # Detect market regime
        vix = market_data.get('vix') if market_data else None
        spy_return = market_data.get('spy_return_30d') if market_data else None
        regime = self.regime_detector.detect_regime(vix, spy_return)

        # Get sector
        sector = self.sector_adjuster.get_sector(ticker)
        features.sector = sector.value
        features.market_regime = regime.value

        # Get predictions
        if self._is_trained and SKLEARN_AVAILABLE:
            predictions, confidence, feature_contrib = self._ml_predict(features)
        else:
            predictions, confidence, feature_contrib = self._rule_based_predict(features)

        # Apply regime and sector adjustments
        regime_adj = self.regime_detector.get_regime_adjustments(regime)
        sector_weights = self.sector_adjuster.get_sector_weights(sector)

        # Combine predictions into overall score
        timeframe_scores = {}
        for tf, pred in predictions.items():
            # Adjust by regime
            if pred > 0:
                adjusted = pred * regime_adj['bullish_multiplier']
            else:
                adjusted = pred * regime_adj['bearish_multiplier']

            timeframe_scores[tf] = adjusted

        # Weighted average of timeframes (1m most important)
        weights = {'1w': 0.2, '1m': 0.5, '3m': 0.3}
        ml_score = sum(timeframe_scores[tf] * w for tf, w in weights.items())

        # Normalize to -10 to +10
        ml_score = max(-10, min(10, ml_score))

        # Adjust confidence by regime
        adjusted_confidence = max(0, min(1, confidence + regime_adj['confidence_adjustment']))

        # Generate recommendation
        recommendation = self._generate_recommendation(
            ml_score, adjusted_confidence, regime, sector
        )

        return MLScore(
            ticker=ticker,
            score_date=date.today(),
            ml_score=round(ml_score, 2),
            confidence=round(adjusted_confidence, 3),
            prediction_7d=round(predictions.get('1w', 0), 2),
            prediction_30d=round(predictions.get('1m', 0), 2),
            prediction_90d=round(predictions.get('3m', 0), 2),
            feature_contributions=feature_contrib,
            model_agreement=self._calculate_model_agreement(features) if self._is_trained else 0.8,
            recommendation=recommendation,
            timeframe_scores={k: round(v, 2) for k, v in timeframe_scores.items()}
        )

    def _ml_predict(
        self,
        features: SignalFeatures
    ) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        """Make predictions using trained ML models."""
        X = features.to_array().reshape(1, -1)
        predictions = {}
        all_preds = {}

        for timeframe in self.TIMEFRAMES:
            X_scaled = self.scalers[timeframe].transform(X)

            # Get predictions from each model
            tf_preds = []
            for model_name, model in self.models[timeframe].items():
                pred = model.predict(X_scaled)[0]
                tf_preds.append(pred)
                all_preds[f'{timeframe}_{model_name}'] = pred

            # Ensemble average
            predictions[timeframe] = np.mean(tf_preds)

        # Calculate confidence from model agreement
        all_values = list(all_preds.values())
        confidence = 1 - (np.std(all_values) / (np.abs(np.mean(all_values)) + 1))
        confidence = max(0.3, min(0.95, confidence))

        # Feature contributions (from RF feature importance)
        feature_contrib = {}
        if hasattr(self.models['1m']['rf'], 'feature_importances_'):
            importances = self.models['1m']['rf'].feature_importances_
            feature_names = SignalFeatures.feature_names()
            for name, imp in zip(feature_names, importances):
                if imp > 0.02:  # Only significant features
                    feature_contrib[name] = round(imp, 3)

        return predictions, confidence, feature_contrib

    def _rule_based_predict(
        self,
        features: SignalFeatures
    ) -> Tuple[Dict[str, float], float, Dict[str, float]]:
        """Fallback rule-based predictions when ML not available."""
        # Calculate base score from features
        score = 0
        confidence_factors = []
        contributions = {}

        # Insider signals (high weight)
        insider_signal = 0
        if features.insider_buy_count > 0:
            insider_signal += features.insider_buy_count * 1.5
            contributions['insider_buys'] = features.insider_buy_count * 0.15
        if features.insider_sell_count > 0:
            insider_signal -= features.insider_sell_count * 1.2
            contributions['insider_sells'] = -features.insider_sell_count * 0.12

        if features.ceo_transaction != 0:
            insider_signal += features.ceo_transaction * 3
            contributions['ceo_transaction'] = features.ceo_transaction * 0.3

        if features.multiple_insider_buy:
            insider_signal += 4
            contributions['multiple_insider_buy'] = 0.2

        score += insider_signal
        if abs(insider_signal) > 2:
            confidence_factors.append(0.8)

        # Institutional signals
        inst_signal = 0
        inst_signal += features.top_fund_new_positions * 2
        inst_signal += features.top_fund_increases * 1
        inst_signal -= features.top_fund_decreases * 1.5
        inst_signal -= features.top_fund_exits * 3

        if features.institutional_convergence:
            inst_signal += 3
            contributions['institutional_convergence'] = 0.15

        score += inst_signal
        if abs(inst_signal) > 2:
            confidence_factors.append(0.75)

        # Hiring signals
        hiring_signal = 0
        hiring_signal += features.commercial_jobs * 0.5
        hiring_signal += features.clinical_jobs * 0.3
        hiring_signal += features.senior_hires * 0.8

        if features.total_jobs_30d >= 5:
            hiring_signal += 2
            contributions['hiring_surge'] = 0.1

        score += hiring_signal

        # Sentiment and executive
        if features.glassdoor_sentiment != 0:
            score += features.glassdoor_sentiment * 2
            contributions['sentiment'] = abs(features.glassdoor_sentiment) * 0.1

        if features.executive_departures >= 2:
            score -= 4
            contributions['executive_exodus'] = 0.2
            confidence_factors.append(0.85)

        # Normalize score
        score = max(-10, min(10, score / 2))

        # Calculate predictions for different timeframes
        predictions = {
            '1w': score * 0.3,   # Smaller short-term
            '1m': score * 0.8,   # Primary timeframe
            '3m': score * 0.5,   # Signal decay
        }

        # Calculate confidence
        if confidence_factors:
            confidence = np.mean(confidence_factors)
        else:
            confidence = 0.5

        return predictions, confidence, contributions

    def _calculate_model_agreement(self, features: SignalFeatures) -> float:
        """Calculate agreement between ensemble models."""
        if not self._is_trained:
            return 0.8

        X = features.to_array().reshape(1, -1)
        all_preds = []

        for timeframe in self.TIMEFRAMES:
            X_scaled = self.scalers[timeframe].transform(X)

            for model in self.models[timeframe].values():
                all_preds.append(model.predict(X_scaled)[0])

        # Agreement is inverse of normalized standard deviation
        if len(all_preds) > 1:
            std = np.std(all_preds)
            mean_abs = np.abs(np.mean(all_preds)) + 1
            agreement = 1 - min(1, std / mean_abs)
            return max(0.3, agreement)

        return 0.8

    def _generate_recommendation(
        self,
        score: float,
        confidence: float,
        regime: MarketRegime,
        sector: BiotechSector
    ) -> str:
        """Generate actionable recommendation."""
        # Determine direction
        if score >= 6 and confidence >= 0.7:
            direction = "STRONG BUY"
            action = "Initiate or add to position aggressively"
        elif score >= 3 and confidence >= 0.5:
            direction = "BUY"
            action = "Consider initiating position"
        elif score <= -6 and confidence >= 0.7:
            direction = "STRONG SELL"
            action = "Exit position or establish short"
        elif score <= -3 and confidence >= 0.5:
            direction = "SELL"
            action = "Reduce position"
        else:
            direction = "NEUTRAL"
            action = "Hold current position, monitor for changes"

        # Add regime context
        regime_note = ""
        if regime == MarketRegime.BEAR:
            regime_note = " (Exercise caution in bear market)"
        elif regime == MarketRegime.CRISIS:
            regime_note = " (Extreme caution - market in crisis mode)"
        elif regime == MarketRegime.BULL:
            regime_note = " (Favorable market conditions)"

        # Add sector note for relevant sectors
        sector_note = ""
        if sector == BiotechSector.GENE_THERAPY and score > 0:
            sector_note = " Gene therapy sector: watch for regulatory catalysts."
        elif sector == BiotechSector.RARE_DISEASE and score > 0:
            sector_note = " Rare disease: commercial buildup signals important."

        return f"{direction}: {action}.{regime_note}{sector_note}"

    def save_model(self, path: str):
        """Save trained models to disk."""
        if not self._is_trained:
            logger.warning("No trained model to save")
            return

        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'is_trained': self._is_trained,
        }

        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Saved model to {path}")

    def load_model(self, path: str):
        """Load trained models from disk."""
        if not os.path.exists(path):
            logger.warning(f"Model file not found: {path}")
            return

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self._is_trained = model_data['is_trained']

        logger.info(f"Loaded model from {path}")

    def get_feature_importance(self, timeframe: str = '1m') -> Dict[str, float]:
        """Get feature importances from trained models."""
        if not self._is_trained:
            return {}

        if 'rf' not in self.models.get(timeframe, {}):
            return {}

        rf = self.models[timeframe]['rf']
        if not hasattr(rf, 'feature_importances_'):
            return {}

        importances = rf.feature_importances_
        feature_names = SignalFeatures.feature_names()

        return {
            name: round(imp, 4)
            for name, imp in sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            if imp > 0.01
        }


def generate_synthetic_training_data(n_samples: int = 500) -> List[Tuple[SignalFeatures, Dict[str, float]]]:
    """Generate synthetic training data for model development."""
    np.random.seed(42)
    data = []

    for _ in range(n_samples):
        # Generate features
        features = SignalFeatures(
            insider_buy_count=np.random.poisson(1),
            insider_sell_count=np.random.poisson(2),
            insider_buy_value=np.random.lognormal(11, 2) if np.random.random() > 0.5 else 0,
            insider_sell_value=np.random.lognormal(11, 2) if np.random.random() > 0.3 else 0,
            ceo_transaction=np.random.choice([-1, 0, 0, 0, 1], p=[0.1, 0.5, 0.2, 0.1, 0.1]),
            cfo_transaction=np.random.choice([-1, 0, 0, 0, 1], p=[0.15, 0.45, 0.2, 0.1, 0.1]),
            multiple_insider_buy=np.random.random() < 0.1,
            multiple_insider_sell=np.random.random() < 0.15,
            avg_insider_signal_age=np.random.uniform(1, 60),
            top_fund_new_positions=np.random.poisson(0.3),
            top_fund_increases=np.random.poisson(1),
            top_fund_decreases=np.random.poisson(0.8),
            top_fund_exits=np.random.poisson(0.2),
            institutional_convergence=np.random.random() < 0.1,
            total_jobs_30d=np.random.poisson(5),
            commercial_jobs=np.random.poisson(1),
            clinical_jobs=np.random.poisson(2),
            rd_jobs=np.random.poisson(3),
            senior_hires=np.random.poisson(0.5),
            hiring_velocity=np.random.uniform(0, 3),
            glassdoor_sentiment=np.random.normal(0, 0.3),
            sentiment_trend=np.random.normal(0, 0.1),
            layoff_mentions=np.random.poisson(0.3),
            executive_departures=np.random.poisson(0.2),
            executive_hires=np.random.poisson(0.3),
            market_regime=np.random.choice(['bull', 'neutral', 'bear']),
            sector=np.random.choice(['oncology', 'rare_disease', 'cns', 'gene_therapy']),
            price_momentum_30d=np.random.normal(0, 10),
            volume_ratio=np.random.lognormal(0, 0.3),
        )

        # Generate returns correlated with features
        base_signal = (
            features.insider_buy_count * 2 -
            features.insider_sell_count * 1.5 +
            features.ceo_transaction * 5 +
            features.top_fund_new_positions * 3 -
            features.top_fund_exits * 4 +
            features.commercial_jobs * 1.5 +
            features.glassdoor_sentiment * 3 -
            features.executive_departures * 4
        )

        # Add noise
        returns = {
            '1w': base_signal * 0.5 + np.random.normal(0, 5),
            '1m': base_signal * 1.0 + np.random.normal(0, 10),
            '3m': base_signal * 0.7 + np.random.normal(0, 15),
        }

        data.append((features, returns))

    return data


if __name__ == '__main__':
    # Test ML scorer
    print("Testing ML Signal Scorer")
    print("=" * 50)

    # Generate training data
    training_data = generate_synthetic_training_data(500)
    print(f"Generated {len(training_data)} training samples")

    # Initialize and train scorer
    scorer = MLSignalScorer()

    if SKLEARN_AVAILABLE:
        metrics = scorer.train(training_data)
        print(f"\nTraining metrics: {json.dumps(metrics, indent=2)}")

    # Test scoring
    test_features = SignalFeatures(
        insider_buy_count=3,
        insider_sell_count=0,
        insider_buy_value=250000,
        ceo_transaction=1,
        multiple_insider_buy=True,
        top_fund_new_positions=2,
        commercial_jobs=5,
        glassdoor_sentiment=0.4,
    )

    result = scorer.score('MRNA', test_features)

    print(f"\nTest Score for MRNA:")
    print(f"  ML Score: {result.ml_score:+.2f}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  7-day prediction: {result.prediction_7d:+.1f}%")
    print(f"  30-day prediction: {result.prediction_30d:+.1f}%")
    print(f"  90-day prediction: {result.prediction_90d:+.1f}%")
    print(f"  Model agreement: {result.model_agreement:.1%}")
    print(f"  Recommendation: {result.recommendation}")

    if result.feature_contributions:
        print(f"\nTop feature contributions:")
        for feature, contrib in sorted(result.feature_contributions.items(),
                                       key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {feature}: {contrib:.3f}")
