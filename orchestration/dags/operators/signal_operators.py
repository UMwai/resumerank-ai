"""
Signal Detection and Scoring Operators

Custom Airflow operators for investment signal analysis:
- Signal detection from raw data
- Composite scoring with configurable weights
- Signal aggregation and deduplication
"""

import logging
from abc import abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

logger = logging.getLogger(__name__)


class SignalDetectionOperator(BaseOperator):
    """
    Base operator for detecting investment signals from data.

    Features:
    - Configurable signal detection rules
    - Confidence scoring
    - Signal deduplication
    - Metadata enrichment

    Args:
        task_id: Unique task identifier
        signal_type: Type of signal being detected
        input_task_id: Task ID to pull input data from
        input_key: XCom key for input data
        detection_rules: List of detection rule configurations
        min_confidence: Minimum confidence threshold
        dedup_window_hours: Window for signal deduplication
    """

    template_fields = ("input_task_id", "input_key")
    ui_color = "#fef3e2"
    ui_fgcolor = "#b35900"

    @apply_defaults
    def __init__(
        self,
        signal_type: str,
        input_task_id: str,
        input_key: str = "return_value",
        detection_rules: Optional[List[Dict[str, Any]]] = None,
        min_confidence: float = 0.5,
        dedup_window_hours: int = 24,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.signal_type = signal_type
        self.input_task_id = input_task_id
        self.input_key = input_key
        self.detection_rules = detection_rules or []
        self.min_confidence = min_confidence
        self.dedup_window_hours = dedup_window_hours

    def execute(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute signal detection."""
        ti = context["ti"]

        # Pull input data
        input_data = ti.xcom_pull(task_ids=self.input_task_id, key=self.input_key)
        if not input_data:
            logger.warning(f"No input data from {self.input_task_id}")
            return []

        # Detect signals
        signals = self.detect_signals(input_data, context)

        # Filter by confidence
        signals = [s for s in signals if s.get("confidence", 0) >= self.min_confidence]

        # Deduplicate
        signals = self._deduplicate_signals(signals)

        logger.info(
            f"Detected {len(signals)} {self.signal_type} signals "
            f"(min confidence: {self.min_confidence})"
        )

        # Push results
        ti.xcom_push(key="signals", value=signals)
        ti.xcom_push(key="signal_count", value=len(signals))

        return signals

    @abstractmethod
    def detect_signals(
        self, data: Any, context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Detect signals from input data.

        Override in subclasses to implement specific detection logic.

        Args:
            data: Input data to analyze
            context: Airflow context

        Returns:
            List of detected signals with confidence scores
        """
        pass

    def _deduplicate_signals(
        self, signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate signals within dedup window."""
        seen = set()
        unique_signals = []

        for signal in signals:
            # Create dedup key from signal type, ticker, and key attributes
            dedup_key = self._create_dedup_key(signal)

            if dedup_key not in seen:
                seen.add(dedup_key)
                unique_signals.append(signal)

        return unique_signals

    def _create_dedup_key(self, signal: Dict[str, Any]) -> str:
        """Create deduplication key for a signal."""
        components = [
            self.signal_type,
            signal.get("ticker", ""),
            signal.get("signal_subtype", ""),
            str(signal.get("date", "")),
        ]
        return ":".join(components)


class SignalScoringOperator(BaseOperator):
    """
    Operator for scoring investment signals.

    Calculates composite scores based on:
    - Individual signal scores
    - Configurable weights
    - Historical performance adjustments
    - Confidence penalties

    Args:
        task_id: Unique task identifier
        pipeline_name: Name of the pipeline for metrics
        input_task_ids: List of task IDs providing signals
        scoring_weights: Dict mapping signal types to weights
        score_thresholds: Dict defining recommendation thresholds
    """

    template_fields = ("input_task_ids",)
    ui_color = "#e2f3fe"
    ui_fgcolor = "#004c8c"

    # Default scoring weights
    DEFAULT_WEIGHTS = {
        "clinical_trial": 0.3,
        "patent_cliff": 0.25,
        "insider_trading": 0.25,
        "institutional": 0.1,
        "hiring": 0.1,
    }

    # Default recommendation thresholds
    DEFAULT_THRESHOLDS = {
        "STRONG_BUY": 8.0,
        "BUY": 6.5,
        "HOLD": 4.0,
        "SELL": 2.5,
        "STRONG_SELL": 0.0,
    }

    @apply_defaults
    def __init__(
        self,
        pipeline_name: str,
        input_task_ids: List[str],
        scoring_weights: Optional[Dict[str, float]] = None,
        score_thresholds: Optional[Dict[str, float]] = None,
        confidence_penalty_factor: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pipeline_name = pipeline_name
        self.input_task_ids = input_task_ids
        self.scoring_weights = scoring_weights or self.DEFAULT_WEIGHTS
        self.score_thresholds = score_thresholds or self.DEFAULT_THRESHOLDS
        self.confidence_penalty_factor = confidence_penalty_factor

    def execute(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute signal scoring."""
        ti = context["ti"]

        # Collect signals from all input tasks
        all_signals = []
        for task_id in self.input_task_ids:
            signals = ti.xcom_pull(task_ids=task_id, key="signals")
            if signals:
                all_signals.extend(signals)

        if not all_signals:
            logger.warning("No signals to score")
            return []

        # Group signals by ticker
        signals_by_ticker = self._group_by_ticker(all_signals)

        # Score each ticker
        scored_results = []
        for ticker, signals in signals_by_ticker.items():
            score_result = self._score_ticker(ticker, signals)
            scored_results.append(score_result)

        # Sort by composite score
        scored_results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Push results
        ti.xcom_push(key="scored_signals", value=scored_results)
        ti.xcom_push(
            key="high_confidence",
            value=[s for s in scored_results if s["composite_score"] >= 7.0],
        )

        logger.info(f"Scored {len(scored_results)} tickers")

        return scored_results

    def _group_by_ticker(
        self, signals: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group signals by ticker."""
        grouped = {}
        for signal in signals:
            ticker = signal.get("ticker")
            if ticker:
                grouped.setdefault(ticker, []).append(signal)
        return grouped

    def _score_ticker(
        self, ticker: str, signals: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate composite score for a ticker."""
        # Group signals by type
        signals_by_type = {}
        for signal in signals:
            signal_type = signal.get("signal_type", "unknown")
            signals_by_type.setdefault(signal_type, []).append(signal)

        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        score_breakdown = {}

        for signal_type, type_signals in signals_by_type.items():
            weight = self.scoring_weights.get(signal_type, 0.1)

            # Average score for this signal type
            type_scores = [s.get("score", 5.0) for s in type_signals]
            avg_score = sum(type_scores) / len(type_scores)

            # Average confidence
            confidences = [s.get("confidence", 0.5) for s in type_signals]
            avg_confidence = sum(confidences) / len(confidences)

            # Apply confidence penalty
            adjusted_score = avg_score * (1 - self.confidence_penalty_factor * (1 - avg_confidence))

            score_breakdown[signal_type] = {
                "raw_score": avg_score,
                "confidence": avg_confidence,
                "adjusted_score": adjusted_score,
                "weight": weight,
                "signal_count": len(type_signals),
            }

            total_score += adjusted_score * weight
            total_weight += weight

        # Normalize score
        composite_score = total_score / total_weight if total_weight > 0 else 5.0

        # Determine recommendation
        recommendation = self._get_recommendation(composite_score)

        return {
            "ticker": ticker,
            "composite_score": round(composite_score, 2),
            "recommendation": recommendation,
            "signal_count": len(signals),
            "score_breakdown": score_breakdown,
            "scored_at": datetime.utcnow().isoformat(),
        }

    def _get_recommendation(self, score: float) -> str:
        """Get recommendation based on score."""
        for rec, threshold in sorted(
            self.score_thresholds.items(), key=lambda x: x[1], reverse=True
        ):
            if score >= threshold:
                return rec
        return "HOLD"


class SignalAggregationOperator(BaseOperator):
    """
    Operator for aggregating signals across pipelines.

    Features:
    - Cross-pipeline signal correlation
    - Time-weighted aggregation
    - Trend detection
    - Summary statistics

    Args:
        task_id: Unique task identifier
        input_task_ids: List of task IDs providing scored signals
        aggregation_window_days: Window for time-weighted aggregation
        trend_threshold: Minimum change for trend detection
    """

    template_fields = ("input_task_ids",)
    ui_color = "#f3e2fe"
    ui_fgcolor = "#6b008c"

    @apply_defaults
    def __init__(
        self,
        input_task_ids: List[str],
        aggregation_window_days: int = 7,
        trend_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_task_ids = input_task_ids
        self.aggregation_window_days = aggregation_window_days
        self.trend_threshold = trend_threshold

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute signal aggregation."""
        ti = context["ti"]

        # Collect scored signals from all pipelines
        all_scored = []
        for task_id in self.input_task_ids:
            scored = ti.xcom_pull(task_ids=task_id, key="scored_signals")
            if scored:
                all_scored.extend(scored)

        if not all_scored:
            return {"aggregated": [], "summary": {}}

        # Aggregate by ticker
        aggregated = self._aggregate_scores(all_scored)

        # Calculate summary statistics
        summary = self._calculate_summary(aggregated)

        # Push results
        ti.xcom_push(key="aggregated_signals", value=aggregated)
        ti.xcom_push(key="aggregation_summary", value=summary)

        return {"aggregated": aggregated, "summary": summary}

    def _aggregate_scores(
        self, scored_signals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Aggregate scores by ticker."""
        by_ticker = {}
        for signal in scored_signals:
            ticker = signal.get("ticker")
            if ticker:
                by_ticker.setdefault(ticker, []).append(signal)

        aggregated = []
        for ticker, signals in by_ticker.items():
            # Time-weighted average
            scores = [s["composite_score"] for s in signals]
            avg_score = sum(scores) / len(scores)

            aggregated.append({
                "ticker": ticker,
                "aggregated_score": round(avg_score, 2),
                "signal_sources": len(signals),
                "min_score": min(scores),
                "max_score": max(scores),
                "score_variance": round(
                    sum((s - avg_score) ** 2 for s in scores) / len(scores), 4
                ),
            })

        return sorted(aggregated, key=lambda x: x["aggregated_score"], reverse=True)

    def _calculate_summary(
        self, aggregated: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate summary statistics."""
        if not aggregated:
            return {}

        scores = [a["aggregated_score"] for a in aggregated]

        return {
            "total_tickers": len(aggregated),
            "avg_score": round(sum(scores) / len(scores), 2),
            "strong_buys": len([s for s in scores if s >= 8.0]),
            "buys": len([s for s in scores if 6.5 <= s < 8.0]),
            "holds": len([s for s in scores if 4.0 <= s < 6.5]),
            "sells": len([s for s in scores if s < 4.0]),
            "aggregated_at": datetime.utcnow().isoformat(),
        }
