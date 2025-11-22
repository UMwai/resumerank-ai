"""Database models for Clinical Trial Signal Detection System."""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from typing import Optional, List, Dict, Any

from psycopg2 import extras

from .connection import get_db_connection

logger = logging.getLogger(__name__)


@dataclass
class Company:
    """Company model representing a biotech company."""
    ticker: str
    company_name: str
    market_cap: Optional[int] = None
    current_price: Optional[float] = None
    sector: str = "Biotechnology"
    cik: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    def save(self) -> None:
        """Insert or update company in database."""
        db = get_db_connection()
        query = """
            INSERT INTO companies (ticker, company_name, market_cap, current_price, sector, cik)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker) DO UPDATE SET
                company_name = EXCLUDED.company_name,
                market_cap = EXCLUDED.market_cap,
                current_price = EXCLUDED.current_price,
                sector = EXCLUDED.sector,
                cik = EXCLUDED.cik
        """
        db.execute(query, (
            self.ticker, self.company_name, self.market_cap,
            self.current_price, self.sector, self.cik
        ))
        logger.debug(f"Saved company: {self.ticker}")

    @classmethod
    def get_by_ticker(cls, ticker: str) -> Optional["Company"]:
        """Retrieve company by ticker."""
        db = get_db_connection()
        result = db.execute(
            "SELECT * FROM companies WHERE ticker = %s", (ticker,)
        )
        if result:
            return cls(**result[0])
        return None

    @classmethod
    def get_all(cls) -> List["Company"]:
        """Retrieve all companies."""
        db = get_db_connection()
        results = db.execute("SELECT * FROM companies ORDER BY ticker")
        return [cls(**row) for row in results]


@dataclass
class Trial:
    """Trial model representing a clinical trial."""
    trial_id: str  # NCT number
    company_ticker: Optional[str] = None
    drug_name: Optional[str] = None
    indication: Optional[str] = None
    phase: Optional[str] = None
    enrollment_target: Optional[int] = None
    enrollment_current: Optional[int] = None
    start_date: Optional[date] = None
    expected_completion: Optional[date] = None
    primary_completion_date: Optional[date] = None
    primary_endpoint: Optional[str] = None
    status: Optional[str] = None
    sponsor: Optional[str] = None
    study_type: Optional[str] = None
    last_updated: Optional[datetime] = None
    created_at: Optional[datetime] = None
    raw_data: Optional[Dict] = None

    def save(self) -> None:
        """Insert or update trial in database."""
        db = get_db_connection()
        query = """
            INSERT INTO trials (
                trial_id, company_ticker, drug_name, indication, phase,
                enrollment_target, enrollment_current, start_date,
                expected_completion, primary_completion_date, primary_endpoint,
                status, sponsor, study_type, last_updated, raw_data
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (trial_id) DO UPDATE SET
                company_ticker = EXCLUDED.company_ticker,
                drug_name = EXCLUDED.drug_name,
                indication = EXCLUDED.indication,
                phase = EXCLUDED.phase,
                enrollment_target = EXCLUDED.enrollment_target,
                enrollment_current = EXCLUDED.enrollment_current,
                start_date = EXCLUDED.start_date,
                expected_completion = EXCLUDED.expected_completion,
                primary_completion_date = EXCLUDED.primary_completion_date,
                primary_endpoint = EXCLUDED.primary_endpoint,
                status = EXCLUDED.status,
                sponsor = EXCLUDED.sponsor,
                study_type = EXCLUDED.study_type,
                last_updated = EXCLUDED.last_updated,
                raw_data = EXCLUDED.raw_data
        """
        raw_data_json = json.dumps(self.raw_data) if self.raw_data else None
        db.execute(query, (
            self.trial_id, self.company_ticker, self.drug_name, self.indication,
            self.phase, self.enrollment_target, self.enrollment_current,
            self.start_date, self.expected_completion, self.primary_completion_date,
            self.primary_endpoint, self.status, self.sponsor, self.study_type,
            self.last_updated, raw_data_json
        ))
        logger.debug(f"Saved trial: {self.trial_id}")

    @classmethod
    def get_by_id(cls, trial_id: str) -> Optional["Trial"]:
        """Retrieve trial by NCT ID."""
        db = get_db_connection()
        result = db.execute(
            "SELECT * FROM trials WHERE trial_id = %s", (trial_id,)
        )
        if result:
            return cls(**result[0])
        return None

    @classmethod
    def get_all(cls) -> List["Trial"]:
        """Retrieve all trials."""
        db = get_db_connection()
        results = db.execute("SELECT * FROM trials ORDER BY trial_id")
        return [cls(**row) for row in results]

    @classmethod
    def get_monitored(cls) -> List["Trial"]:
        """Retrieve all monitored (active) trials."""
        db = get_db_connection()
        results = db.execute("""
            SELECT * FROM trials
            WHERE status NOT IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN')
            ORDER BY expected_completion ASC
        """)
        return [cls(**row) for row in results]


@dataclass
class TrialSignal:
    """Signal detected for a clinical trial."""
    trial_id: str
    signal_type: str
    signal_value: Optional[str] = None
    signal_weight: int = 0
    detected_date: Optional[datetime] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    raw_data: Optional[Dict] = None
    processed: bool = False
    signal_id: Optional[int] = None

    def save(self) -> int:
        """Insert signal into database and return signal_id."""
        db = get_db_connection()
        query = """
            INSERT INTO trial_signals (
                trial_id, signal_type, signal_value, signal_weight,
                detected_date, source, source_url, raw_data, processed
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING signal_id
        """
        raw_data_json = json.dumps(self.raw_data) if self.raw_data else None
        detected = self.detected_date or datetime.now()
        result = db.execute(query, (
            self.trial_id, self.signal_type, self.signal_value,
            self.signal_weight, detected, self.source, self.source_url,
            raw_data_json, self.processed
        ))
        if result:
            self.signal_id = result[0]["signal_id"]
            logger.info(f"Saved signal: {self.signal_type} for {self.trial_id}")
            return self.signal_id
        return 0

    @classmethod
    def get_by_trial(cls, trial_id: str, days: int = 30) -> List["TrialSignal"]:
        """Get recent signals for a trial."""
        db = get_db_connection()
        results = db.execute("""
            SELECT * FROM trial_signals
            WHERE trial_id = %s
            AND detected_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY detected_date DESC
        """, (trial_id, days))
        return [cls(**row) for row in results]

    @classmethod
    def get_recent(cls, days: int = 1) -> List["TrialSignal"]:
        """Get all recent signals."""
        db = get_db_connection()
        results = db.execute("""
            SELECT ts.*, t.drug_name, c.company_name
            FROM trial_signals ts
            JOIN trials t ON ts.trial_id = t.trial_id
            LEFT JOIN companies c ON t.company_ticker = c.ticker
            WHERE ts.detected_date >= CURRENT_DATE - INTERVAL '%s days'
            ORDER BY ts.detected_date DESC
        """, (days,))
        return results  # Return raw dicts for email digest


@dataclass
class TrialScore:
    """Composite score for a trial."""
    trial_id: str
    composite_score: float
    confidence: float
    recommendation: str
    score_date: Optional[date] = None
    contributing_signals: Optional[Dict] = None
    score_id: Optional[int] = None
    created_at: Optional[datetime] = None

    def save(self) -> None:
        """Insert or update trial score."""
        db = get_db_connection()
        query = """
            INSERT INTO trial_scores (
                trial_id, score_date, composite_score, confidence,
                recommendation, contributing_signals
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (trial_id, score_date) DO UPDATE SET
                composite_score = EXCLUDED.composite_score,
                confidence = EXCLUDED.confidence,
                recommendation = EXCLUDED.recommendation,
                contributing_signals = EXCLUDED.contributing_signals
        """
        signals_json = json.dumps(self.contributing_signals) if self.contributing_signals else None
        score_dt = self.score_date or date.today()
        db.execute(query, (
            self.trial_id, score_dt, self.composite_score,
            self.confidence, self.recommendation, signals_json
        ))
        logger.debug(f"Saved score: {self.composite_score:.2f} for {self.trial_id}")

    @classmethod
    def get_latest(cls, trial_id: str) -> Optional["TrialScore"]:
        """Get the latest score for a trial."""
        db = get_db_connection()
        result = db.execute("""
            SELECT * FROM trial_scores
            WHERE trial_id = %s
            ORDER BY score_date DESC
            LIMIT 1
        """, (trial_id,))
        if result:
            return cls(**result[0])
        return None

    @classmethod
    def get_actionable(cls) -> List[Dict]:
        """Get trials with actionable scores (>=7 or <=3)."""
        db = get_db_connection()
        return db.execute("""
            SELECT ts.*, t.drug_name, t.indication, c.ticker, c.company_name
            FROM trial_scores ts
            JOIN trials t ON ts.trial_id = t.trial_id
            LEFT JOIN companies c ON t.company_ticker = c.ticker
            WHERE ts.score_date = CURRENT_DATE
            AND (ts.composite_score >= 7 OR ts.composite_score <= 3)
            ORDER BY ts.composite_score DESC
        """)


@dataclass
class SECFiling:
    """SEC EDGAR filing record."""
    company_ticker: str
    filing_type: str
    filing_date: date
    accession_number: str
    filing_url: Optional[str] = None
    description: Optional[str] = None
    raw_content: Optional[str] = None
    processed: bool = False
    filing_id: Optional[int] = None
    created_at: Optional[datetime] = None

    def save(self) -> None:
        """Insert SEC filing into database."""
        db = get_db_connection()
        query = """
            INSERT INTO sec_filings (
                company_ticker, filing_type, filing_date, accession_number,
                filing_url, description, raw_content, processed
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (accession_number) DO NOTHING
        """
        db.execute(query, (
            self.company_ticker, self.filing_type, self.filing_date,
            self.accession_number, self.filing_url, self.description,
            self.raw_content, self.processed
        ))
        logger.debug(f"Saved SEC filing: {self.accession_number}")

    @classmethod
    def exists(cls, accession_number: str) -> bool:
        """Check if filing already exists."""
        db = get_db_connection()
        result = db.execute(
            "SELECT 1 FROM sec_filings WHERE accession_number = %s",
            (accession_number,)
        )
        return len(result) > 0

    @classmethod
    def get_unprocessed(cls) -> List["SECFiling"]:
        """Get unprocessed filings."""
        db = get_db_connection()
        results = db.execute("""
            SELECT * FROM sec_filings
            WHERE processed = FALSE
            ORDER BY filing_date DESC
        """)
        return [cls(**row) for row in results]


@dataclass
class TrialHistory:
    """Historical snapshot of trial data for change detection."""
    trial_id: str
    enrollment_current: Optional[int] = None
    status: Optional[str] = None
    expected_completion: Optional[date] = None
    primary_endpoint: Optional[str] = None
    raw_data: Optional[Dict] = None
    snapshot_date: Optional[datetime] = None
    history_id: Optional[int] = None

    def save(self) -> None:
        """Save a history snapshot."""
        db = get_db_connection()
        query = """
            INSERT INTO trial_history (
                trial_id, enrollment_current, status, expected_completion,
                primary_endpoint, raw_data
            )
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        raw_data_json = json.dumps(self.raw_data) if self.raw_data else None
        db.execute(query, (
            self.trial_id, self.enrollment_current, self.status,
            self.expected_completion, self.primary_endpoint, raw_data_json
        ))

    @classmethod
    def get_latest(cls, trial_id: str) -> Optional["TrialHistory"]:
        """Get the most recent history snapshot for a trial."""
        db = get_db_connection()
        result = db.execute("""
            SELECT * FROM trial_history
            WHERE trial_id = %s
            ORDER BY snapshot_date DESC
            LIMIT 1
        """, (trial_id,))
        if result:
            return cls(**result[0])
        return None
