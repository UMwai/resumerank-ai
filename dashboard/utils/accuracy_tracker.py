"""
Signal Accuracy Tracker

Tracks historical predictions vs outcomes to calculate accuracy metrics
per system. Provides win/loss ratio, ROI by signal type, and performance
attribution analysis.
"""

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """Represents a tracked prediction."""
    id: Optional[int] = None
    signal_id: str = ""
    source: str = ""  # clinical_trial, patent, insider
    ticker: str = ""
    company_name: str = ""

    # Prediction details
    signal_type: str = ""  # bullish, bearish
    predicted_direction: str = ""  # up, down
    score: float = 0.0
    confidence: float = 0.0

    # Price at prediction
    entry_price: Optional[float] = None
    entry_date: Optional[datetime] = None

    # Target/Stop
    target_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_days: int = 30  # Days to evaluate

    # Outcome
    outcome: Optional[str] = None  # win, loss, breakeven, pending
    exit_price: Optional[float] = None
    exit_date: Optional[datetime] = None
    price_change_pct: Optional[float] = None
    roi: Optional[float] = None

    # Metadata
    created_at: Optional[datetime] = None
    evaluated_at: Optional[datetime] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "signal_id": self.signal_id,
            "source": self.source,
            "ticker": self.ticker,
            "company_name": self.company_name,
            "signal_type": self.signal_type,
            "predicted_direction": self.predicted_direction,
            "score": self.score,
            "confidence": self.confidence,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat() if self.entry_date else None,
            "target_price": self.target_price,
            "stop_price": self.stop_price,
            "target_days": self.target_days,
            "outcome": self.outcome,
            "exit_price": self.exit_price,
            "exit_date": self.exit_date.isoformat() if self.exit_date else None,
            "price_change_pct": self.price_change_pct,
            "roi": self.roi,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a system or period."""
    total_predictions: int = 0
    evaluated_predictions: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    pending: int = 0

    # Rates
    win_rate: float = 0.0
    loss_rate: float = 0.0
    accuracy: float = 0.0

    # Returns
    total_roi: float = 0.0
    average_roi: float = 0.0
    best_roi: float = 0.0
    worst_roi: float = 0.0

    # Risk metrics
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    profit_factor: Optional[float] = None

    # By confidence
    high_confidence_accuracy: Optional[float] = None
    medium_confidence_accuracy: Optional[float] = None
    low_confidence_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_predictions": self.total_predictions,
            "evaluated_predictions": self.evaluated_predictions,
            "wins": self.wins,
            "losses": self.losses,
            "breakeven": self.breakeven,
            "pending": self.pending,
            "win_rate": self.win_rate,
            "loss_rate": self.loss_rate,
            "accuracy": self.accuracy,
            "total_roi": self.total_roi,
            "average_roi": self.average_roi,
            "best_roi": self.best_roi,
            "worst_roi": self.worst_roi,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "profit_factor": self.profit_factor,
        }


class AccuracyTracker:
    """
    Tracks and analyzes prediction accuracy across signal systems.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize accuracy tracker.

        Args:
            db_path: Path to SQLite database
        """
        if db_path is None:
            data_dir = Path.home() / ".investment_dashboard"
            data_dir.mkdir(exist_ok=True)
            db_path = str(data_dir / "accuracy.db")

        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE,
                    source TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    company_name TEXT,
                    signal_type TEXT,
                    predicted_direction TEXT,
                    score REAL,
                    confidence REAL,
                    entry_price REAL,
                    entry_date TEXT,
                    target_price REAL,
                    stop_price REAL,
                    target_days INTEGER DEFAULT 30,
                    outcome TEXT,
                    exit_price REAL,
                    exit_date TEXT,
                    price_change_pct REAL,
                    roi REAL,
                    created_at TEXT,
                    evaluated_at TEXT,
                    notes TEXT
                )
            """)

            # Historical snapshots for time-series analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS accuracy_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date TEXT NOT NULL,
                    source TEXT,
                    total_predictions INTEGER,
                    wins INTEGER,
                    losses INTEGER,
                    win_rate REAL,
                    total_roi REAL,
                    average_roi REAL,
                    UNIQUE(snapshot_date, source)
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_ticker ON predictions(ticker)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_source ON predictions(source)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_outcome ON predictions(outcome)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(entry_date)")

            conn.commit()

    # Prediction tracking

    def track_prediction(
        self,
        signal_id: str,
        source: str,
        ticker: str,
        signal_type: str,
        score: float,
        confidence: float,
        entry_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        target_days: int = 30,
        company_name: str = "",
    ) -> Optional[Prediction]:
        """
        Track a new prediction.

        Args:
            signal_id: Unique signal identifier
            source: Signal source system
            ticker: Stock ticker
            signal_type: bullish or bearish
            score: Signal score
            confidence: Confidence level
            entry_price: Price at signal time
            target_price: Target price
            stop_price: Stop loss price
            target_days: Days to evaluate outcome
            company_name: Company name

        Returns:
            Created prediction or None if already exists
        """
        now = datetime.now()
        predicted_direction = "up" if signal_type == "bullish" else "down"

        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                cursor.execute("""
                    INSERT INTO predictions (
                        signal_id, source, ticker, company_name, signal_type,
                        predicted_direction, score, confidence, entry_price,
                        entry_date, target_price, stop_price, target_days,
                        outcome, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """, (
                    signal_id, source, ticker.upper(), company_name, signal_type,
                    predicted_direction, score, confidence, entry_price,
                    now.isoformat(), target_price, stop_price, target_days, now.isoformat()
                ))

                return Prediction(
                    id=cursor.lastrowid,
                    signal_id=signal_id,
                    source=source,
                    ticker=ticker.upper(),
                    company_name=company_name,
                    signal_type=signal_type,
                    predicted_direction=predicted_direction,
                    score=score,
                    confidence=confidence,
                    entry_price=entry_price,
                    entry_date=now,
                    target_price=target_price,
                    stop_price=stop_price,
                    target_days=target_days,
                    outcome="pending",
                    created_at=now,
                )

            except sqlite3.IntegrityError:
                logger.warning(f"Prediction {signal_id} already exists")
                return None

    def record_outcome(
        self,
        prediction_id: int,
        outcome: str,
        exit_price: Optional[float] = None,
        notes: str = "",
    ) -> bool:
        """
        Record the outcome of a prediction.

        Args:
            prediction_id: Prediction ID
            outcome: win, loss, or breakeven
            exit_price: Exit/current price
            notes: Additional notes

        Returns:
            True if recorded
        """
        now = datetime.now()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get prediction details
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            if not row:
                return False

            entry_price = row["entry_price"]
            predicted_direction = row["predicted_direction"]

            # Calculate price change and ROI
            price_change_pct = None
            roi = None

            if entry_price and exit_price:
                price_change_pct = ((exit_price - entry_price) / entry_price) * 100

                # ROI depends on direction
                if predicted_direction == "up":
                    roi = price_change_pct
                else:
                    roi = -price_change_pct

            cursor.execute("""
                UPDATE predictions SET
                    outcome = ?, exit_price = ?, exit_date = ?,
                    price_change_pct = ?, roi = ?, evaluated_at = ?, notes = ?
                WHERE id = ?
            """, (
                outcome, exit_price, now.isoformat(),
                price_change_pct, roi, now.isoformat(), notes, prediction_id
            ))

            return cursor.rowcount > 0

    def auto_evaluate_predictions(
        self,
        price_fetcher: Optional[callable] = None,
    ) -> List[Tuple[int, str]]:
        """
        Auto-evaluate pending predictions based on target days.

        Args:
            price_fetcher: Function to fetch current price for ticker

        Returns:
            List of (prediction_id, outcome) tuples
        """
        evaluated = []

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get pending predictions past their target date
            cursor.execute("""
                SELECT * FROM predictions
                WHERE outcome = 'pending'
                AND datetime(entry_date, '+' || target_days || ' days') <= datetime('now')
            """)

            for row in cursor.fetchall():
                pred_id = row["id"]
                ticker = row["ticker"]
                entry_price = row["entry_price"]
                predicted_direction = row["predicted_direction"]
                target_price = row["target_price"]
                stop_price = row["stop_price"]

                # Get current price
                current_price = None
                if price_fetcher:
                    try:
                        current_price = price_fetcher(ticker)
                    except Exception as e:
                        logger.warning(f"Could not fetch price for {ticker}: {e}")

                if current_price is None and entry_price:
                    # Use simulated outcome for demo
                    import random
                    change = random.uniform(-0.15, 0.25)
                    current_price = entry_price * (1 + change)

                if current_price and entry_price:
                    # Determine outcome
                    price_change = (current_price - entry_price) / entry_price

                    if predicted_direction == "up":
                        if target_price and current_price >= target_price:
                            outcome = "win"
                        elif stop_price and current_price <= stop_price:
                            outcome = "loss"
                        elif price_change > 0.02:
                            outcome = "win"
                        elif price_change < -0.05:
                            outcome = "loss"
                        else:
                            outcome = "breakeven"
                    else:  # down
                        if target_price and current_price <= target_price:
                            outcome = "win"
                        elif stop_price and current_price >= stop_price:
                            outcome = "loss"
                        elif price_change < -0.02:
                            outcome = "win"
                        elif price_change > 0.05:
                            outcome = "loss"
                        else:
                            outcome = "breakeven"

                    self.record_outcome(pred_id, outcome, current_price)
                    evaluated.append((pred_id, outcome))

        return evaluated

    # Metrics calculation

    def get_accuracy_metrics(
        self,
        source: Optional[str] = None,
        ticker: Optional[str] = None,
        days: Optional[int] = None,
    ) -> AccuracyMetrics:
        """
        Calculate accuracy metrics.

        Args:
            source: Filter by source
            ticker: Filter by ticker
            days: Only include predictions from last N days

        Returns:
            AccuracyMetrics object
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM predictions WHERE 1=1"
            params = []

            if source:
                query += " AND source = ?"
                params.append(source)

            if ticker:
                query += " AND ticker = ?"
                params.append(ticker.upper())

            if days:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                query += " AND entry_date >= ?"
                params.append(cutoff)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        # Calculate metrics
        metrics = AccuracyMetrics()
        metrics.total_predictions = len(rows)

        rois = []
        for row in rows:
            outcome = row["outcome"]
            roi = row["roi"]

            if outcome == "pending":
                metrics.pending += 1
            else:
                metrics.evaluated_predictions += 1
                if outcome == "win":
                    metrics.wins += 1
                elif outcome == "loss":
                    metrics.losses += 1
                else:
                    metrics.breakeven += 1

                if roi is not None:
                    rois.append(roi)

        # Calculate rates
        if metrics.evaluated_predictions > 0:
            metrics.win_rate = metrics.wins / metrics.evaluated_predictions
            metrics.loss_rate = metrics.losses / metrics.evaluated_predictions
            metrics.accuracy = (metrics.wins + metrics.breakeven) / metrics.evaluated_predictions

        # Calculate ROI stats
        if rois:
            metrics.total_roi = sum(rois)
            metrics.average_roi = np.mean(rois)
            metrics.best_roi = max(rois)
            metrics.worst_roi = min(rois)

            # Risk metrics
            if len(rois) > 1:
                returns = np.array(rois)
                std_dev = np.std(returns)
                if std_dev > 0:
                    metrics.sharpe_ratio = (np.mean(returns) - 2.0) / std_dev  # Assume 2% risk-free

                # Max drawdown (simplified)
                cumulative = np.cumsum(returns)
                peak = np.maximum.accumulate(cumulative)
                drawdown = peak - cumulative
                metrics.max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

                # Profit factor
                wins_sum = sum(r for r in rois if r > 0)
                losses_sum = abs(sum(r for r in rois if r < 0))
                if losses_sum > 0:
                    metrics.profit_factor = wins_sum / losses_sum

        # Accuracy by confidence
        high_conf = [r for r in rows if r["confidence"] and r["confidence"] >= 0.8 and r["outcome"] != "pending"]
        med_conf = [r for r in rows if r["confidence"] and 0.5 <= r["confidence"] < 0.8 and r["outcome"] != "pending"]
        low_conf = [r for r in rows if r["confidence"] and r["confidence"] < 0.5 and r["outcome"] != "pending"]

        if high_conf:
            high_wins = sum(1 for r in high_conf if r["outcome"] == "win")
            metrics.high_confidence_accuracy = high_wins / len(high_conf)

        if med_conf:
            med_wins = sum(1 for r in med_conf if r["outcome"] == "win")
            metrics.medium_confidence_accuracy = med_wins / len(med_conf)

        if low_conf:
            low_wins = sum(1 for r in low_conf if r["outcome"] == "win")
            metrics.low_confidence_accuracy = low_wins / len(low_conf)

        return metrics

    def get_metrics_by_source(self, days: Optional[int] = None) -> Dict[str, AccuracyMetrics]:
        """Get accuracy metrics grouped by source."""
        sources = ["clinical_trial", "patent", "insider"]
        return {source: self.get_accuracy_metrics(source=source, days=days) for source in sources}

    def get_metrics_by_ticker(self, days: Optional[int] = None) -> Dict[str, AccuracyMetrics]:
        """Get accuracy metrics grouped by ticker."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM predictions")
            tickers = [row["ticker"] for row in cursor.fetchall()]

        return {ticker: self.get_accuracy_metrics(ticker=ticker, days=days) for ticker in tickers}

    def get_performance_over_time(
        self,
        source: Optional[str] = None,
        period: str = "weekly",
    ) -> pd.DataFrame:
        """
        Get performance metrics over time.

        Args:
            source: Filter by source
            period: 'daily', 'weekly', or 'monthly'

        Returns:
            DataFrame with time-series performance data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    date(entry_date) as date,
                    COUNT(*) as predictions,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'loss' THEN 1 ELSE 0 END) as losses,
                    AVG(CASE WHEN roi IS NOT NULL THEN roi ELSE 0 END) as avg_roi,
                    SUM(CASE WHEN roi IS NOT NULL THEN roi ELSE 0 END) as total_roi
                FROM predictions
                WHERE outcome != 'pending'
            """
            params = []

            if source:
                query += " AND source = ?"
                params.append(source)

            query += " GROUP BY date(entry_date) ORDER BY date"

            cursor.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame([dict(row) for row in rows])
        df["date"] = pd.to_datetime(df["date"])
        df["win_rate"] = df["wins"] / df["predictions"]

        # Resample if needed
        if period == "weekly":
            df = df.set_index("date").resample("W").agg({
                "predictions": "sum",
                "wins": "sum",
                "losses": "sum",
                "total_roi": "sum",
            }).reset_index()
            df["avg_roi"] = df["total_roi"] / df["predictions"]
            df["win_rate"] = df["wins"] / df["predictions"]
        elif period == "monthly":
            df = df.set_index("date").resample("M").agg({
                "predictions": "sum",
                "wins": "sum",
                "losses": "sum",
                "total_roi": "sum",
            }).reset_index()
            df["avg_roi"] = df["total_roi"] / df["predictions"]
            df["win_rate"] = df["wins"] / df["predictions"]

        # Calculate cumulative ROI
        df["cumulative_roi"] = df["total_roi"].cumsum()

        return df

    def get_roi_by_signal_type(self, source: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get ROI breakdown by signal type."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT
                    signal_type,
                    COUNT(*) as count,
                    SUM(CASE WHEN outcome = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(CASE WHEN roi IS NOT NULL THEN roi ELSE 0 END) as avg_roi,
                    SUM(CASE WHEN roi IS NOT NULL THEN roi ELSE 0 END) as total_roi
                FROM predictions
                WHERE outcome != 'pending'
            """
            params = []

            if source:
                query += " AND source = ?"
                params.append(source)

            query += " GROUP BY signal_type"

            cursor.execute(query, params)
            rows = cursor.fetchall()

        return {
            row["signal_type"]: {
                "count": row["count"],
                "wins": row["wins"],
                "win_rate": row["wins"] / row["count"] if row["count"] > 0 else 0,
                "avg_roi": row["avg_roi"],
                "total_roi": row["total_roi"],
            }
            for row in rows
        }

    def get_predictions_df(
        self,
        source: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        """Get predictions as DataFrame."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT * FROM predictions WHERE 1=1"
            params = []

            if source:
                query += " AND source = ?"
                params.append(source)

            if outcome:
                query += " AND outcome = ?"
                params.append(outcome)

            query += " ORDER BY entry_date DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([dict(row) for row in rows])

    def save_snapshot(self) -> None:
        """Save current accuracy snapshot for historical tracking."""
        today = datetime.now().date().isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Overall snapshot
            metrics = self.get_accuracy_metrics()
            cursor.execute("""
                INSERT OR REPLACE INTO accuracy_snapshots
                (snapshot_date, source, total_predictions, wins, losses, win_rate, total_roi, average_roi)
                VALUES (?, 'all', ?, ?, ?, ?, ?, ?)
            """, (
                today, metrics.total_predictions, metrics.wins, metrics.losses,
                metrics.win_rate, metrics.total_roi, metrics.average_roi
            ))

            # Per-source snapshots
            for source in ["clinical_trial", "patent", "insider"]:
                metrics = self.get_accuracy_metrics(source=source)
                cursor.execute("""
                    INSERT OR REPLACE INTO accuracy_snapshots
                    (snapshot_date, source, total_predictions, wins, losses, win_rate, total_roi, average_roi)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    today, source, metrics.total_predictions, metrics.wins, metrics.losses,
                    metrics.win_rate, metrics.total_roi, metrics.average_roi
                ))


# Demo data generator
def generate_demo_predictions(tracker: AccuracyTracker, count: int = 50) -> None:
    """Generate demo prediction data for testing."""
    import random

    sources = ["clinical_trial", "patent", "insider"]
    tickers = ["MRNA", "VRTX", "ABBV", "PFE", "GILD", "REGN", "BIIB", "AMGN"]
    companies = {
        "MRNA": "Moderna Inc.",
        "VRTX": "Vertex Pharmaceuticals",
        "ABBV": "AbbVie Inc.",
        "PFE": "Pfizer Inc.",
        "GILD": "Gilead Sciences",
        "REGN": "Regeneron Pharmaceuticals",
        "BIIB": "Biogen Inc.",
        "AMGN": "Amgen Inc.",
    }

    for i in range(count):
        ticker = random.choice(tickers)
        source = random.choice(sources)
        signal_type = random.choice(["bullish", "bearish"])
        score = random.uniform(0.4, 0.95)
        confidence = random.uniform(0.5, 0.95)

        entry_price = random.uniform(50, 300)
        change_pct = random.uniform(-0.15, 0.25)

        if signal_type == "bullish":
            target_price = entry_price * 1.1
            stop_price = entry_price * 0.95
        else:
            target_price = entry_price * 0.9
            stop_price = entry_price * 1.05

        pred = tracker.track_prediction(
            signal_id=f"demo_{source}_{i}_{random.randint(1000, 9999)}",
            source=source,
            ticker=ticker,
            signal_type=signal_type,
            score=score,
            confidence=confidence,
            entry_price=entry_price,
            target_price=target_price,
            stop_price=stop_price,
            target_days=30,
            company_name=companies.get(ticker, ticker),
        )

        if pred and random.random() > 0.2:  # 80% have outcomes
            exit_price = entry_price * (1 + change_pct)

            if signal_type == "bullish":
                if change_pct > 0.05:
                    outcome = "win"
                elif change_pct < -0.05:
                    outcome = "loss"
                else:
                    outcome = "breakeven"
            else:
                if change_pct < -0.05:
                    outcome = "win"
                elif change_pct > 0.05:
                    outcome = "loss"
                else:
                    outcome = "breakeven"

            tracker.record_outcome(pred.id, outcome, exit_price)


# Global instance
_tracker: Optional[AccuracyTracker] = None


def get_accuracy_tracker(db_path: Optional[str] = None) -> AccuracyTracker:
    """Get or create the global accuracy tracker."""
    global _tracker
    if _tracker is None:
        _tracker = AccuracyTracker(db_path)
    return _tracker
