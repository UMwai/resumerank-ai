-- Initialize PostgreSQL databases for Investment Signals
--
-- This script runs when the PostgreSQL container is first created.
-- It creates the investment_signals database and user.

-- Create signals user if not exists
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'signals') THEN
        CREATE ROLE signals WITH LOGIN PASSWORD 'signals_password';
    END IF;
END
$$;

-- Create investment_signals database
SELECT 'CREATE DATABASE investment_signals OWNER signals'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'investment_signals')\gexec

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE investment_signals TO signals;

-- Connect to investment_signals database and create schema
\c investment_signals

-- Create schema for each pipeline
CREATE SCHEMA IF NOT EXISTS clinical_trials;
CREATE SCHEMA IF NOT EXISTS patent_intelligence;
CREATE SCHEMA IF NOT EXISTS insider_hiring;
CREATE SCHEMA IF NOT EXISTS orchestration;

-- Grant schema access
GRANT ALL ON SCHEMA clinical_trials TO signals;
GRANT ALL ON SCHEMA patent_intelligence TO signals;
GRANT ALL ON SCHEMA insider_hiring TO signals;
GRANT ALL ON SCHEMA orchestration TO signals;

-- Clinical Trials Tables
CREATE TABLE IF NOT EXISTS clinical_trials.trials (
    trial_id VARCHAR(50) PRIMARY KEY,
    nct_id VARCHAR(20) UNIQUE,
    title TEXT,
    sponsor VARCHAR(255),
    company_ticker VARCHAR(10),
    phase VARCHAR(20),
    status VARCHAR(50),
    condition TEXT,
    intervention TEXT,
    start_date DATE,
    completion_date DATE,
    primary_endpoint TEXT,
    enrollment INTEGER,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clinical_trials.signals (
    signal_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(50) REFERENCES clinical_trials.trials(trial_id),
    signal_type VARCHAR(50),
    signal_value TEXT,
    signal_weight FLOAT,
    source VARCHAR(50),
    source_url TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_data JSONB
);

CREATE TABLE IF NOT EXISTS clinical_trials.scores (
    score_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(50) REFERENCES clinical_trials.trials(trial_id),
    composite_score FLOAT,
    recommendation VARCHAR(20),
    confidence FLOAT,
    scoring_date DATE,
    score_breakdown JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patent Intelligence Tables
CREATE TABLE IF NOT EXISTS patent_intelligence.drugs (
    drug_id SERIAL PRIMARY KEY,
    brand_name VARCHAR(255),
    generic_name VARCHAR(255),
    active_ingredient TEXT,
    branded_company VARCHAR(255),
    branded_company_ticker VARCHAR(10),
    therapeutic_area VARCHAR(100),
    annual_revenue_millions FLOAT,
    anda_count INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patent_intelligence.patents (
    patent_id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES patent_intelligence.drugs(drug_id),
    patent_number VARCHAR(20),
    patent_type VARCHAR(50),
    base_expiration_date DATE,
    adjusted_expiration_date DATE,
    final_expiration_date DATE,
    pta_days INTEGER DEFAULT 0,
    pte_days INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patent_intelligence.calendar_events (
    event_id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES patent_intelligence.drugs(drug_id),
    event_type VARCHAR(50),
    event_date DATE,
    event_description TEXT,
    priority VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insider/Hiring Tables
CREATE TABLE IF NOT EXISTS insider_hiring.form4_filings (
    filing_id SERIAL PRIMARY KEY,
    accession_number VARCHAR(30) UNIQUE,
    ticker VARCHAR(10),
    company_name VARCHAR(255),
    insider_name VARCHAR(255),
    insider_title VARCHAR(100),
    transaction_type VARCHAR(20),
    transaction_date DATE,
    shares_traded BIGINT,
    price_per_share FLOAT,
    total_value FLOAT,
    ownership_type VARCHAR(20),
    filing_date TIMESTAMP,
    source_url TEXT,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS insider_hiring.form13f_holdings (
    holding_id SERIAL PRIMARY KEY,
    manager_name VARCHAR(255),
    manager_cik VARCHAR(20),
    ticker VARCHAR(10),
    company_name VARCHAR(255),
    shares_held BIGINT,
    value_thousands FLOAT,
    report_date DATE,
    quarter VARCHAR(10),
    change_shares BIGINT,
    change_percent FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS insider_hiring.job_postings (
    posting_id SERIAL PRIMARY KEY,
    company_name VARCHAR(255),
    ticker VARCHAR(10),
    job_title VARCHAR(255),
    department VARCHAR(100),
    location VARCHAR(255),
    job_type VARCHAR(50),
    seniority_level VARCHAR(50),
    posted_date DATE,
    source VARCHAR(50),
    source_url TEXT,
    is_leadership BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS insider_hiring.scores (
    score_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10),
    composite_score FLOAT,
    insider_score FLOAT,
    institutional_score FLOAT,
    hiring_score FLOAT,
    recommendation VARCHAR(20),
    confidence FLOAT,
    signal_count INTEGER,
    scoring_date DATE,
    score_breakdown JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orchestration Tables
CREATE TABLE IF NOT EXISTS orchestration.pipeline_runs (
    run_id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100),
    dag_id VARCHAR(100),
    run_type VARCHAR(50),
    status VARCHAR(20),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds FLOAT,
    records_processed INTEGER,
    signals_found INTEGER,
    errors INTEGER,
    error_messages TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS orchestration.alerts_sent (
    alert_id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50),
    priority VARCHAR(20),
    channel VARCHAR(20),
    recipient TEXT,
    subject TEXT,
    message TEXT,
    signal_data JSONB,
    dedup_key VARCHAR(64),
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success BOOLEAN DEFAULT TRUE
);

CREATE TABLE IF NOT EXISTS orchestration.metrics (
    metric_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    labels JSONB,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_trials_ticker ON clinical_trials.trials(company_ticker);
CREATE INDEX IF NOT EXISTS idx_trials_status ON clinical_trials.trials(status);
CREATE INDEX IF NOT EXISTS idx_trials_last_updated ON clinical_trials.trials(last_updated);

CREATE INDEX IF NOT EXISTS idx_signals_trial_id ON clinical_trials.signals(trial_id);
CREATE INDEX IF NOT EXISTS idx_signals_type ON clinical_trials.signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_signals_detected_at ON clinical_trials.signals(detected_at);

CREATE INDEX IF NOT EXISTS idx_form4_ticker ON insider_hiring.form4_filings(ticker);
CREATE INDEX IF NOT EXISTS idx_form4_filing_date ON insider_hiring.form4_filings(filing_date);

CREATE INDEX IF NOT EXISTS idx_form13f_ticker ON insider_hiring.form13f_holdings(ticker);
CREATE INDEX IF NOT EXISTS idx_form13f_report_date ON insider_hiring.form13f_holdings(report_date);

CREATE INDEX IF NOT EXISTS idx_jobs_ticker ON insider_hiring.job_postings(ticker);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_date ON insider_hiring.job_postings(posted_date);

CREATE INDEX IF NOT EXISTS idx_patents_drug_id ON patent_intelligence.patents(drug_id);
CREATE INDEX IF NOT EXISTS idx_patents_expiration ON patent_intelligence.patents(final_expiration_date);

CREATE INDEX IF NOT EXISTS idx_alerts_dedup ON orchestration.alerts_sent(dedup_key);
CREATE INDEX IF NOT EXISTS idx_alerts_sent_at ON orchestration.alerts_sent(sent_at);

-- Grant table access
GRANT ALL ON ALL TABLES IN SCHEMA clinical_trials TO signals;
GRANT ALL ON ALL TABLES IN SCHEMA patent_intelligence TO signals;
GRANT ALL ON ALL TABLES IN SCHEMA insider_hiring TO signals;
GRANT ALL ON ALL TABLES IN SCHEMA orchestration TO signals;

GRANT ALL ON ALL SEQUENCES IN SCHEMA clinical_trials TO signals;
GRANT ALL ON ALL SEQUENCES IN SCHEMA patent_intelligence TO signals;
GRANT ALL ON ALL SEQUENCES IN SCHEMA insider_hiring TO signals;
GRANT ALL ON ALL SEQUENCES IN SCHEMA orchestration TO signals;

-- Output completion message
DO $$ BEGIN RAISE NOTICE 'Database initialization complete'; END $$;
