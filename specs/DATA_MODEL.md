# Investment Intelligence Platform - Data Model

## Overview

This document defines the data models and schemas for the Investment Intelligence Platform, covering clinical trial signals, patent intelligence, insider activity, portfolio management, and trading system state.

---

## Core Domain Models

### Clinical Trial Signals

```python
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional, List, Dict
from enum import Enum

class TrialPhase(Enum):
    PHASE_1 = "Phase 1"
    PHASE_1_2 = "Phase 1/Phase 2"
    PHASE_2 = "Phase 2"
    PHASE_2_3 = "Phase 2/Phase 3"
    PHASE_3 = "Phase 3"
    PHASE_4 = "Phase 4"
    NA = "N/A"

class TrialStatus(Enum):
    NOT_YET_RECRUITING = "Not yet recruiting"
    RECRUITING = "Recruiting"
    ACTIVE_NOT_RECRUITING = "Active, not recruiting"
    COMPLETED = "Completed"
    SUSPENDED = "Suspended"
    TERMINATED = "Terminated"
    WITHDRAWN = "Withdrawn"

class SignalType(Enum):
    AMENDMENT = "amendment"
    ENROLLMENT = "enrollment"
    TIMELINE = "timeline"
    PUBLICATION = "publication"
    REGULATORY = "regulatory"

class Recommendation(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

@dataclass
class ClinicalTrial:
    """Clinical trial information from ClinicalTrials.gov"""
    nct_id: str                     # Primary key (NCT12345678)
    title: str
    brief_summary: str
    phase: TrialPhase
    status: TrialStatus
    sponsor: str
    ticker: Optional[str]           # Company stock ticker
    company_name: str
    conditions: List[str]
    interventions: List[Dict]
    primary_outcomes: List[Dict]
    enrollment: int
    enrollment_type: str            # Actual or Anticipated
    start_date: Optional[date]
    completion_date: Optional[date]
    last_update_date: datetime
    locations: List[Dict]
    created_at: datetime
    updated_at: datetime

@dataclass
class TrialAmendment:
    """Detected protocol amendment"""
    id: str
    nct_id: str
    amendment_date: date
    detected_date: datetime
    change_type: str                # endpoint, enrollment, timeline, etc.
    field_changed: str
    old_value: str
    new_value: str
    significance_score: float       # 0-10 signal strength
    analysis_notes: Optional[str]

@dataclass
class ClinicalTrialSignal:
    """Scored clinical trial signal"""
    id: str
    nct_id: str
    ticker: str
    company: str
    trial_title: str
    signal_type: SignalType
    signal_score: float             # 0-10
    confidence: float               # 0-1
    detected_at: datetime
    description: str
    recommendation: Recommendation
    expected_catalyst_date: Optional[date]
    details: Dict
    is_active: bool
    acknowledged: bool
```

### Patent/IP Intelligence

```python
class PatentStatus(Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    PENDING = "pending"
    ABANDONED = "abandoned"

class ExclusivityType(Enum):
    PATENT = "patent"
    ORPHAN_DRUG = "orphan_drug"
    PEDIATRIC = "pediatric"
    NEW_CHEMICAL = "new_chemical"
    DATA_EXCLUSIVITY = "data_exclusivity"

@dataclass
class Patent:
    """Patent information from USPTO"""
    patent_number: str              # Primary key
    title: str
    abstract: str
    filing_date: date
    issue_date: date
    expiry_date: date
    assignee: str
    ticker: Optional[str]
    inventors: List[str]
    claims_count: int
    citations_count: int
    patent_family: List[str]
    status: PatentStatus
    drug_names: List[str]
    created_at: datetime

@dataclass
class OrangeBookEntry:
    """FDA Orange Book exclusivity data"""
    id: str
    drug_name: str
    active_ingredient: str
    nda_number: str
    applicant: str
    ticker: Optional[str]
    patent_numbers: List[str]
    exclusivity_type: ExclusivityType
    exclusivity_expiry: date
    approval_date: date
    strength: str
    dosage_form: str
    route: str

@dataclass
class PatentCliffSignal:
    """Patent cliff investment signal"""
    id: str
    drug_name: str
    ticker: str
    company: str
    annual_revenue_mm: float
    patent_expiry_date: date
    months_to_expiry: int
    generic_filers: List[str]
    expected_revenue_loss_pct: float
    signal_score: float             # 0-10
    confidence: float               # 0-1
    trade_recommendation: Dict      # Suggested trade structure
    detected_at: datetime
    is_active: bool
```

### Insider Activity

```python
class TransactionType(Enum):
    PURCHASE = "P"
    SALE = "S"
    AWARD = "A"
    EXERCISE = "M"
    GIFT = "G"

class InsiderRole(Enum):
    CEO = "CEO"
    CFO = "CFO"
    COO = "COO"
    DIRECTOR = "Director"
    VP = "VP"
    TEN_PCT_OWNER = "10% Owner"
    OTHER = "Other"

@dataclass
class Form4Filing:
    """SEC Form 4 insider transaction"""
    id: str
    accession_number: str
    ticker: str
    company_name: str
    insider_name: str
    insider_title: str
    insider_role: InsiderRole
    transaction_date: date
    transaction_type: TransactionType
    shares: int
    price: float
    value: float
    shares_owned_after: int
    is_direct: bool
    footnotes: Optional[str]
    filed_at: datetime
    created_at: datetime

@dataclass
class ThirteenFHolding:
    """13F institutional holding"""
    id: str
    cik: str
    institution_name: str
    ticker: str
    company_name: str
    shares: int
    value: float
    shares_change: int
    shares_change_pct: float
    report_date: date
    filed_at: datetime
    is_new_position: bool
    is_closed_position: bool

@dataclass
class HiringSignal:
    """LinkedIn hiring activity signal"""
    id: str
    ticker: str
    company_name: str
    job_title: str
    department: str
    seniority_level: str
    location: str
    posted_date: date
    is_clinical: bool               # Clinical trial related
    is_regulatory: bool             # FDA/regulatory related
    is_commercial: bool             # Sales/marketing
    detected_at: datetime

@dataclass
class InsiderSignal:
    """Combined insider activity signal"""
    id: str
    ticker: str
    company: str
    signal_type: str                # form4, 13f, hiring, cluster
    signal_score: float             # 0-10
    confidence: float               # 0-1
    detected_at: datetime
    description: str
    recommendation: Recommendation
    details: Dict
    transactions: List[Form4Filing]
    is_active: bool
```

### Portfolio Management

```python
class PositionDirection(Enum):
    LONG = "long"
    SHORT = "short"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Position:
    """Current portfolio position"""
    id: str
    ticker: str
    direction: PositionDirection
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    weight_pct: float
    entry_date: date
    signal_source: str              # clinical_trial, patent, insider
    signal_id: Optional[str]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    last_updated: datetime

@dataclass
class Trade:
    """Executed trade"""
    id: str
    ticker: str
    direction: PositionDirection
    shares: int
    price: float
    value: float
    commission: float
    order_type: OrderType
    executed_at: datetime
    signal_source: str
    signal_id: Optional[str]
    notes: Optional[str]

@dataclass
class Portfolio:
    """Portfolio state"""
    id: str
    name: str
    positions: List[Position]
    cash: float
    total_value: float
    invested_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_pnl: float
    inception_date: date
    last_rebalance: datetime
    created_at: datetime
    updated_at: datetime

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    portfolio_id: str
    period: str                     # 1d, 1w, 1m, 3m, ytd, 1y, all
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    beta: float
    alpha: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    calculated_at: datetime
```

### Market Regime

```python
class MarketRegime(Enum):
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    UNKNOWN = "UNKNOWN"

@dataclass
class RegimeState:
    """Current market regime detection"""
    id: str
    regime: MarketRegime
    confidence: float               # 0-1
    detected_at: datetime
    indicators: Dict                # SMA, VIX, trend
    allocation_adjustments: Dict    # Recommended allocations
    duration_days: int

@dataclass
class RegimeTransition:
    """Regime change event"""
    id: str
    old_regime: MarketRegime
    new_regime: MarketRegime
    transition_date: datetime
    confidence: float
    trigger_indicators: Dict
    recommended_actions: List[str]
```

---

## Database Schemas

### PostgreSQL Schema - Clinical Trials

```sql
-- Clinical trials table
CREATE TABLE clinical_trials (
    nct_id VARCHAR(15) PRIMARY KEY,
    title TEXT NOT NULL,
    brief_summary TEXT,
    phase VARCHAR(20),
    status VARCHAR(50),
    sponsor VARCHAR(255),
    ticker VARCHAR(10),
    company_name VARCHAR(255),
    conditions TEXT[],
    interventions JSONB,
    primary_outcomes JSONB,
    enrollment INTEGER,
    enrollment_type VARCHAR(20),
    start_date DATE,
    completion_date DATE,
    last_update_date TIMESTAMP,
    locations JSONB,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trials_ticker ON clinical_trials(ticker);
CREATE INDEX idx_trials_phase ON clinical_trials(phase);
CREATE INDEX idx_trials_status ON clinical_trials(status);
CREATE INDEX idx_trials_sponsor ON clinical_trials(sponsor);

-- Trial amendments
CREATE TABLE trial_amendments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id VARCHAR(15) NOT NULL REFERENCES clinical_trials(nct_id),
    amendment_date DATE NOT NULL,
    detected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    change_type VARCHAR(50) NOT NULL,
    field_changed VARCHAR(100),
    old_value TEXT,
    new_value TEXT,
    significance_score DOUBLE PRECISION,
    analysis_notes TEXT
);

CREATE INDEX idx_amendments_trial ON trial_amendments(nct_id);
CREATE INDEX idx_amendments_date ON trial_amendments(amendment_date);

-- Clinical trial signals
CREATE TABLE clinical_trial_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id VARCHAR(15) NOT NULL REFERENCES clinical_trials(nct_id),
    ticker VARCHAR(10),
    company VARCHAR(255),
    trial_title TEXT,
    signal_type VARCHAR(30) NOT NULL,
    signal_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    recommendation VARCHAR(20),
    expected_catalyst_date DATE,
    details JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100)
);

CREATE INDEX idx_ct_signals_ticker ON clinical_trial_signals(ticker);
CREATE INDEX idx_ct_signals_score ON clinical_trial_signals(signal_score);
CREATE INDEX idx_ct_signals_active ON clinical_trial_signals(is_active);
```

### PostgreSQL Schema - Patent Intelligence

```sql
-- Patents table
CREATE TABLE patents (
    patent_number VARCHAR(20) PRIMARY KEY,
    title TEXT NOT NULL,
    abstract TEXT,
    filing_date DATE,
    issue_date DATE,
    expiry_date DATE,
    assignee VARCHAR(255),
    ticker VARCHAR(10),
    inventors TEXT[],
    claims_count INTEGER,
    citations_count INTEGER,
    patent_family TEXT[],
    status VARCHAR(20),
    drug_names TEXT[],
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_patents_ticker ON patents(ticker);
CREATE INDEX idx_patents_expiry ON patents(expiry_date);
CREATE INDEX idx_patents_assignee ON patents(assignee);

-- Orange Book exclusivities
CREATE TABLE orange_book_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    drug_name VARCHAR(255) NOT NULL,
    active_ingredient VARCHAR(255),
    nda_number VARCHAR(20),
    applicant VARCHAR(255),
    ticker VARCHAR(10),
    patent_numbers TEXT[],
    exclusivity_type VARCHAR(50),
    exclusivity_expiry DATE,
    approval_date DATE,
    strength VARCHAR(100),
    dosage_form VARCHAR(100),
    route VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_orange_book_ticker ON orange_book_entries(ticker);
CREATE INDEX idx_orange_book_expiry ON orange_book_entries(exclusivity_expiry);

-- Patent cliff signals
CREATE TABLE patent_cliff_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    drug_name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10),
    company VARCHAR(255),
    annual_revenue_mm DOUBLE PRECISION,
    patent_expiry_date DATE NOT NULL,
    months_to_expiry INTEGER,
    generic_filers TEXT[],
    expected_revenue_loss_pct DOUBLE PRECISION,
    signal_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    trade_recommendation JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_patent_signals_ticker ON patent_cliff_signals(ticker);
CREATE INDEX idx_patent_signals_expiry ON patent_cliff_signals(patent_expiry_date);
```

### PostgreSQL Schema - Insider Activity

```sql
-- Form 4 filings
CREATE TABLE form4_filings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    accession_number VARCHAR(30) NOT NULL UNIQUE,
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    insider_name VARCHAR(255) NOT NULL,
    insider_title VARCHAR(255),
    insider_role VARCHAR(30),
    transaction_date DATE NOT NULL,
    transaction_type CHAR(1) NOT NULL,
    shares INTEGER NOT NULL,
    price DOUBLE PRECISION,
    value DOUBLE PRECISION,
    shares_owned_after BIGINT,
    is_direct BOOLEAN,
    footnotes TEXT,
    filed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_form4_ticker ON form4_filings(ticker);
CREATE INDEX idx_form4_date ON form4_filings(transaction_date);
CREATE INDEX idx_form4_type ON form4_filings(transaction_type);
CREATE INDEX idx_form4_insider ON form4_filings(insider_name);

-- 13F holdings
CREATE TABLE thirteenf_holdings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cik VARCHAR(20) NOT NULL,
    institution_name VARCHAR(255) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    shares BIGINT NOT NULL,
    value BIGINT,
    shares_change INTEGER,
    shares_change_pct DOUBLE PRECISION,
    report_date DATE NOT NULL,
    filed_at TIMESTAMP,
    is_new_position BOOLEAN,
    is_closed_position BOOLEAN,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_13f_ticker ON thirteenf_holdings(ticker);
CREATE INDEX idx_13f_institution ON thirteenf_holdings(institution_name);
CREATE INDEX idx_13f_report_date ON thirteenf_holdings(report_date);

-- Hiring signals
CREATE TABLE hiring_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10),
    company_name VARCHAR(255) NOT NULL,
    job_title VARCHAR(255) NOT NULL,
    department VARCHAR(100),
    seniority_level VARCHAR(50),
    location VARCHAR(255),
    posted_date DATE,
    is_clinical BOOLEAN,
    is_regulatory BOOLEAN,
    is_commercial BOOLEAN,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_hiring_ticker ON hiring_signals(ticker);
CREATE INDEX idx_hiring_date ON hiring_signals(posted_date);

-- Combined insider signals
CREATE TABLE insider_signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker VARCHAR(10) NOT NULL,
    company VARCHAR(255),
    signal_type VARCHAR(30) NOT NULL,
    signal_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT,
    recommendation VARCHAR(20),
    details JSONB,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX idx_insider_signals_ticker ON insider_signals(ticker);
CREATE INDEX idx_insider_signals_score ON insider_signals(signal_score);
```

### PostgreSQL Schema - Portfolio

```sql
-- Portfolios
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL,
    cash DOUBLE PRECISION DEFAULT 0,
    total_value DOUBLE PRECISION DEFAULT 0,
    invested_value DOUBLE PRECISION DEFAULT 0,
    unrealized_pnl DOUBLE PRECISION DEFAULT 0,
    realized_pnl DOUBLE PRECISION DEFAULT 0,
    inception_date DATE NOT NULL,
    last_rebalance TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    ticker VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    shares INTEGER NOT NULL,
    avg_cost DOUBLE PRECISION NOT NULL,
    current_price DOUBLE PRECISION,
    market_value DOUBLE PRECISION,
    unrealized_pnl DOUBLE PRECISION,
    weight_pct DOUBLE PRECISION,
    entry_date DATE NOT NULL,
    signal_source VARCHAR(50),
    signal_id UUID,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_positions_portfolio ON positions(portfolio_id);
CREATE INDEX idx_positions_ticker ON positions(ticker);

-- Trades
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    ticker VARCHAR(10) NOT NULL,
    direction VARCHAR(10) NOT NULL,
    shares INTEGER NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    commission DOUBLE PRECISION DEFAULT 0,
    order_type VARCHAR(20),
    executed_at TIMESTAMP NOT NULL,
    signal_source VARCHAR(50),
    signal_id UUID,
    notes TEXT
);

CREATE INDEX idx_trades_portfolio ON trades(portfolio_id);
CREATE INDEX idx_trades_ticker ON trades(ticker);
CREATE INDEX idx_trades_date ON trades(executed_at);

-- Performance snapshots
CREATE TABLE performance_snapshots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id),
    snapshot_date DATE NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    daily_return DOUBLE PRECISION,
    cumulative_return DOUBLE PRECISION,
    drawdown DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_perf_portfolio ON performance_snapshots(portfolio_id);
CREATE INDEX idx_perf_date ON performance_snapshots(snapshot_date);
```

### TimescaleDB Schema - Market Data

```sql
-- OHLCV price data (TimescaleDB hypertable)
CREATE TABLE ohlcv_daily (
    time TIMESTAMPTZ NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    open DOUBLE PRECISION,
    high DOUBLE PRECISION,
    low DOUBLE PRECISION,
    close DOUBLE PRECISION,
    volume BIGINT,
    adjusted_close DOUBLE PRECISION,
    PRIMARY KEY (time, ticker)
);

SELECT create_hypertable('ohlcv_daily', 'time');

CREATE INDEX idx_ohlcv_ticker ON ohlcv_daily(ticker, time DESC);

-- Add compression policy
ALTER TABLE ohlcv_daily SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'ticker'
);

SELECT add_compression_policy('ohlcv_daily', INTERVAL '30 days');

-- Regime state history
CREATE TABLE regime_states (
    time TIMESTAMPTZ NOT NULL,
    regime VARCHAR(20) NOT NULL,
    confidence DOUBLE PRECISION,
    sma_50 DOUBLE PRECISION,
    sma_200 DOUBLE PRECISION,
    vix DOUBLE PRECISION,
    indicators JSONB,
    PRIMARY KEY (time)
);

SELECT create_hypertable('regime_states', 'time');

-- Portfolio value history
CREATE TABLE portfolio_history (
    time TIMESTAMPTZ NOT NULL,
    portfolio_id UUID NOT NULL,
    total_value DOUBLE PRECISION NOT NULL,
    cash DOUBLE PRECISION,
    invested DOUBLE PRECISION,
    daily_pnl DOUBLE PRECISION,
    PRIMARY KEY (time, portfolio_id)
);

SELECT create_hypertable('portfolio_history', 'time');
```

---

## Data Flow Patterns

### Signal Generation Flow

```
Data Source → Ingestion → Storage → Analysis → Signal → Alert
     │            │          │         │         │       │
ClinicalTrials→ Parse   → trials → Amendment → Score → Email
     API        JSON       table    Detection   0-10    Slack
```

### Portfolio Update Flow

```
Signal → Validation → Sizing → Order → Execution → Position
   │          │          │        │         │          │
Score > 7   Risk     Kelly   Market/    Fill     Update
           Check    Criterion  Limit           Portfolio
```

### Regime Detection Flow

```
Market Data → Indicators → Classification → Allocation
     │            │             │              │
  OHLCV      SMA 50/200      BULL/BEAR     80/20
  + VIX      Calculate       Detect        Equity/Cash
```

---

## Naming Conventions

### Tables

| Pattern | Example | Description |
|---------|---------|-------------|
| `{entity}` | `patents`, `portfolios` | Primary entity tables |
| `{entity}_signals` | `clinical_trial_signals` | Signal tables |
| `{entity}_history` | `portfolio_history` | Time-series history |
| `{action}_log` | `trade_log` | Action audit tables |

### Columns

| Pattern | Example | Description |
|---------|---------|-------------|
| `{entity}_id` | `portfolio_id`, `signal_id` | Foreign keys |
| `{action}_at` | `detected_at`, `executed_at` | Timestamps |
| `is_{adjective}` | `is_active`, `is_direct` | Boolean flags |
| `{entity}_pct` | `weight_pct`, `change_pct` | Percentages |

### Indices

| Pattern | Example | Description |
|---------|---------|-------------|
| `idx_{table}_{column}` | `idx_trades_ticker` | Single column |
| `idx_{table}_{col1}_{col2}` | `idx_ohlcv_ticker_time` | Composite |

---

## Data Retention Policies

| Data Type | Retention | Compression |
|-----------|-----------|-------------|
| Raw OHLCV ticks | 7 days | None |
| 1-minute OHLCV | 30 days | After 7 days |
| Daily OHLCV | 10 years | After 30 days |
| Signals | 2 years | None |
| Trades | Forever | None |
| Alerts | 90 days | None |
| Audit logs | 7 years | After 30 days |
