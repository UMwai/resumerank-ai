-- Insider Activity + Hiring Signals System
-- PostgreSQL Database Schema
-- Version 1.0

-- Create extension for UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create database (run as superuser)
-- CREATE DATABASE insider_signals;

-- Companies reference table
CREATE TABLE IF NOT EXISTS companies (
    company_id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    company_name VARCHAR(300),
    cik VARCHAR(20),  -- SEC CIK number
    sector VARCHAR(100) DEFAULT 'Biotech',
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_companies_ticker ON companies(ticker);
CREATE INDEX idx_companies_cik ON companies(cik);

-- Insider transactions from SEC Form 4
CREATE TABLE IF NOT EXISTS insider_transactions (
    transaction_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) NOT NULL,
    company_cik VARCHAR(20),
    insider_name VARCHAR(200) NOT NULL,
    insider_cik VARCHAR(20),
    insider_title VARCHAR(100),
    is_director BOOLEAN DEFAULT FALSE,
    is_officer BOOLEAN DEFAULT FALSE,
    is_ten_percent_owner BOOLEAN DEFAULT FALSE,
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(30),  -- 'Purchase', 'Sale', 'Option Exercise', etc.
    transaction_code VARCHAR(5),   -- SEC transaction code: P, S, A, D, etc.
    shares INT NOT NULL,
    price_per_share DECIMAL(12,4),
    transaction_value DECIMAL(18,2),
    shares_owned_after BIGINT,
    ownership_nature VARCHAR(20),  -- 'Direct', 'Indirect'
    is_10b5_1_plan BOOLEAN DEFAULT FALSE,
    footnotes TEXT,
    filing_date DATE NOT NULL,
    filing_url VARCHAR(500),
    signal_weight INT DEFAULT 0,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_insider_transaction UNIQUE (company_ticker, insider_name, transaction_date, transaction_type, shares, price_per_share)
);

CREATE INDEX idx_insider_trans_ticker ON insider_transactions(company_ticker);
CREATE INDEX idx_insider_trans_date ON insider_transactions(transaction_date);
CREATE INDEX idx_insider_trans_filing_date ON insider_transactions(filing_date);
CREATE INDEX idx_insider_trans_type ON insider_transactions(transaction_type);
CREATE INDEX idx_insider_trans_value ON insider_transactions(transaction_value);

-- Institutional holdings from SEC 13F
CREATE TABLE IF NOT EXISTS institutional_holdings (
    holding_id SERIAL PRIMARY KEY,
    fund_name VARCHAR(300) NOT NULL,
    fund_cik VARCHAR(20),
    company_ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(300),
    cusip VARCHAR(20),
    quarter_end DATE NOT NULL,
    shares BIGINT NOT NULL,
    market_value BIGINT,  -- Value in thousands
    shares_prev_quarter BIGINT,
    pct_change_shares DECIMAL(10,2),
    is_new_position BOOLEAN DEFAULT FALSE,
    is_exit BOOLEAN DEFAULT FALSE,
    put_call VARCHAR(10),  -- 'Put', 'Call', or NULL
    investment_discretion VARCHAR(20),  -- 'SOLE', 'SHARED', 'NONE'
    filing_date DATE,
    filing_url VARCHAR(500),
    signal_weight INT DEFAULT 0,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_holding UNIQUE (fund_cik, company_ticker, quarter_end)
);

CREATE INDEX idx_holdings_fund ON institutional_holdings(fund_name);
CREATE INDEX idx_holdings_ticker ON institutional_holdings(company_ticker);
CREATE INDEX idx_holdings_quarter ON institutional_holdings(quarter_end);
CREATE INDEX idx_holdings_change ON institutional_holdings(pct_change_shares);

-- Job postings
CREATE TABLE IF NOT EXISTS job_postings (
    job_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(300),
    job_title VARCHAR(300) NOT NULL,
    department VARCHAR(100),  -- 'R&D', 'Commercial', 'Regulatory', 'Manufacturing', 'Clinical', 'Admin'
    seniority_level VARCHAR(50),  -- 'Entry', 'Mid', 'Senior', 'Executive', 'Director', 'VP', 'C-Level'
    location VARCHAR(200),
    employment_type VARCHAR(50),  -- 'Full-time', 'Part-time', 'Contract'
    description TEXT,
    post_date DATE,
    first_seen_date DATE NOT NULL DEFAULT CURRENT_DATE,
    last_seen_date DATE NOT NULL DEFAULT CURRENT_DATE,
    removal_date DATE,
    source VARCHAR(50) NOT NULL,  -- 'LinkedIn', 'Indeed', 'Company Website', etc.
    source_url VARCHAR(500),
    is_senior_role BOOLEAN DEFAULT FALSE,
    is_commercial_role BOOLEAN DEFAULT FALSE,
    is_rd_role BOOLEAN DEFAULT FALSE,
    is_clinical_role BOOLEAN DEFAULT FALSE,
    is_manufacturing_role BOOLEAN DEFAULT FALSE,
    signal_weight INT DEFAULT 0,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_job UNIQUE (company_ticker, job_title, source, post_date)
);

CREATE INDEX idx_jobs_ticker ON job_postings(company_ticker);
CREATE INDEX idx_jobs_department ON job_postings(department);
CREATE INDEX idx_jobs_post_date ON job_postings(post_date);
CREATE INDEX idx_jobs_first_seen ON job_postings(first_seen_date);
CREATE INDEX idx_jobs_removal ON job_postings(removal_date);

-- Executive changes from Form 8-K
CREATE TABLE IF NOT EXISTS executive_changes (
    change_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) NOT NULL,
    company_cik VARCHAR(20),
    executive_name VARCHAR(200) NOT NULL,
    title VARCHAR(150),
    change_type VARCHAR(30) NOT NULL,  -- 'Departure', 'Hire', 'Promotion', 'Role Change'
    effective_date DATE,
    announcement_date DATE NOT NULL,
    reason TEXT,
    is_voluntary BOOLEAN,
    successor_name VARCHAR(200),
    filing_url VARCHAR(500),
    filing_text TEXT,
    ai_analysis JSONB,  -- AI-generated analysis of the departure
    severity_score INT,  -- 1-10 scale
    signal_weight INT DEFAULT 0,
    raw_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_exec_change UNIQUE (company_ticker, executive_name, change_type, announcement_date)
);

CREATE INDEX idx_exec_ticker ON executive_changes(company_ticker);
CREATE INDEX idx_exec_date ON executive_changes(announcement_date);
CREATE INDEX idx_exec_type ON executive_changes(change_type);
CREATE INDEX idx_exec_title ON executive_changes(title);

-- Glassdoor sentiment (for future implementation)
CREATE TABLE IF NOT EXISTS glassdoor_sentiment (
    review_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(300),
    review_date DATE,
    overall_rating DECIMAL(2,1),
    ceo_approval BOOLEAN,
    recommend_to_friend BOOLEAN,
    business_outlook VARCHAR(20),  -- 'Positive', 'Neutral', 'Negative'
    pros TEXT,
    cons TEXT,
    review_text TEXT,
    job_title VARCHAR(200),
    employment_status VARCHAR(50),  -- 'Current', 'Former'
    sentiment_score DECIMAL(4,3),  -- -1.0 to +1.0
    mentions_layoffs BOOLEAN DEFAULT FALSE,
    mentions_pipeline BOOLEAN DEFAULT FALSE,
    mentions_management BOOLEAN DEFAULT FALSE,
    ai_analysis JSONB,
    signal_weight INT DEFAULT 0,
    source_url VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_glassdoor_ticker ON glassdoor_sentiment(company_ticker);
CREATE INDEX idx_glassdoor_date ON glassdoor_sentiment(review_date);
CREATE INDEX idx_glassdoor_rating ON glassdoor_sentiment(overall_rating);

-- Aggregate signal scores per company
CREATE TABLE IF NOT EXISTS signal_scores (
    score_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) NOT NULL,
    score_date DATE NOT NULL,
    composite_score DECIMAL(5,2),  -- -10.0 to +10.0
    confidence DECIMAL(4,3),  -- 0.0 to 1.0
    signal_count INT DEFAULT 0,
    insider_score DECIMAL(5,2) DEFAULT 0,
    institutional_score DECIMAL(5,2) DEFAULT 0,
    hiring_score DECIMAL(5,2) DEFAULT 0,
    sentiment_score DECIMAL(5,2) DEFAULT 0,
    recommendation VARCHAR(50),  -- 'STRONG BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG SELL'
    contributing_signals JSONB,  -- Details of signals that contributed
    lookback_days INT DEFAULT 90,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_score UNIQUE (company_ticker, score_date)
);

CREATE INDEX idx_scores_ticker ON signal_scores(company_ticker);
CREATE INDEX idx_scores_date ON signal_scores(score_date);
CREATE INDEX idx_scores_composite ON signal_scores(composite_score);
CREATE INDEX idx_scores_recommendation ON signal_scores(recommendation);

-- Individual signals (detailed breakdown)
CREATE TABLE IF NOT EXISTS signals (
    signal_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) NOT NULL,
    signal_date DATE NOT NULL,
    signal_category VARCHAR(50) NOT NULL,  -- 'insider', 'institutional', 'hiring', 'sentiment', 'executive'
    signal_type VARCHAR(100) NOT NULL,  -- e.g., 'CEO_PURCHASE', 'FUND_NEW_POSITION', 'COMMERCIAL_HIRING'
    signal_description TEXT,
    raw_weight INT NOT NULL,  -- Original weight from signal definition
    days_ago INT,
    decay_factor DECIMAL(5,4),
    weighted_score DECIMAL(6,3),
    source_id INT,  -- Reference to source table (transaction_id, holding_id, etc.)
    source_table VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_ticker ON signals(company_ticker);
CREATE INDEX idx_signals_date ON signals(signal_date);
CREATE INDEX idx_signals_category ON signals(signal_category);
CREATE INDEX idx_signals_type ON signals(signal_type);

-- Email digest history
CREATE TABLE IF NOT EXISTS email_digests (
    digest_id SERIAL PRIMARY KEY,
    digest_date DATE NOT NULL,
    recipient_count INT,
    top_bullish JSONB,  -- Top 5 bullish companies
    top_bearish JSONB,  -- Top 5 bearish companies
    new_signals_count INT,
    content_html TEXT,
    content_text TEXT,
    sent_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'sent', 'failed'
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_digests_date ON email_digests(digest_date);
CREATE INDEX idx_digests_status ON email_digests(status);

-- Scraper run history (for monitoring)
CREATE TABLE IF NOT EXISTS scraper_runs (
    run_id SERIAL PRIMARY KEY,
    scraper_name VARCHAR(50) NOT NULL,  -- 'form4', '13f', 'jobs', 'glassdoor'
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) DEFAULT 'running',  -- 'running', 'completed', 'failed'
    records_processed INT DEFAULT 0,
    records_inserted INT DEFAULT 0,
    records_updated INT DEFAULT 0,
    errors_count INT DEFAULT 0,
    error_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_scraper_runs_name ON scraper_runs(scraper_name);
CREATE INDEX idx_scraper_runs_start ON scraper_runs(start_time);
CREATE INDEX idx_scraper_runs_status ON scraper_runs(status);

-- Views for common queries

-- View: Recent insider activity (last 30 days)
CREATE OR REPLACE VIEW v_recent_insider_activity AS
SELECT
    it.company_ticker,
    c.company_name,
    it.insider_name,
    it.insider_title,
    it.transaction_type,
    it.transaction_date,
    it.shares,
    it.price_per_share,
    it.transaction_value,
    it.is_10b5_1_plan,
    it.signal_weight,
    CURRENT_DATE - it.transaction_date AS days_ago
FROM insider_transactions it
LEFT JOIN companies c ON it.company_ticker = c.ticker
WHERE it.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
ORDER BY it.transaction_date DESC;

-- View: Institutional position changes (latest quarter)
CREATE OR REPLACE VIEW v_institutional_changes AS
SELECT
    ih.fund_name,
    ih.company_ticker,
    ih.company_name,
    ih.quarter_end,
    ih.shares,
    ih.shares_prev_quarter,
    ih.pct_change_shares,
    ih.is_new_position,
    ih.is_exit,
    ih.signal_weight
FROM institutional_holdings ih
WHERE ih.quarter_end = (SELECT MAX(quarter_end) FROM institutional_holdings)
ORDER BY ih.pct_change_shares DESC;

-- View: Current signal scores with details
CREATE OR REPLACE VIEW v_current_signals AS
SELECT
    ss.company_ticker,
    c.company_name,
    ss.score_date,
    ss.composite_score,
    ss.confidence,
    ss.signal_count,
    ss.insider_score,
    ss.institutional_score,
    ss.hiring_score,
    ss.recommendation
FROM signal_scores ss
LEFT JOIN companies c ON ss.company_ticker = c.ticker
WHERE ss.score_date = (SELECT MAX(score_date) FROM signal_scores)
ORDER BY ss.composite_score DESC;

-- View: Job posting trends
CREATE OR REPLACE VIEW v_job_trends AS
SELECT
    company_ticker,
    COUNT(*) AS total_jobs,
    SUM(CASE WHEN is_commercial_role THEN 1 ELSE 0 END) AS commercial_jobs,
    SUM(CASE WHEN is_rd_role THEN 1 ELSE 0 END) AS rd_jobs,
    SUM(CASE WHEN is_clinical_role THEN 1 ELSE 0 END) AS clinical_jobs,
    SUM(CASE WHEN is_manufacturing_role THEN 1 ELSE 0 END) AS manufacturing_jobs,
    SUM(CASE WHEN is_senior_role THEN 1 ELSE 0 END) AS senior_roles,
    MIN(first_seen_date) AS earliest_posting,
    MAX(first_seen_date) AS latest_posting
FROM job_postings
WHERE removal_date IS NULL  -- Still active
GROUP BY company_ticker
ORDER BY total_jobs DESC;

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for companies table
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Grant permissions (adjust user as needed)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO insider_signals_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO insider_signals_user;

-- Sample data for top biotech-focused funds
INSERT INTO companies (ticker, company_name, sector) VALUES
    ('MRNA', 'Moderna Inc', 'Biotech'),
    ('BNTX', 'BioNTech SE', 'Biotech'),
    ('VRTX', 'Vertex Pharmaceuticals', 'Biotech'),
    ('REGN', 'Regeneron Pharmaceuticals', 'Biotech'),
    ('BIIB', 'Biogen Inc', 'Biotech'),
    ('ALNY', 'Alnylam Pharmaceuticals', 'Biotech'),
    ('BMRN', 'BioMarin Pharmaceutical', 'Biotech'),
    ('INCY', 'Incyte Corporation', 'Biotech'),
    ('IONS', 'Ionis Pharmaceuticals', 'Biotech'),
    ('NBIX', 'Neurocrine Biosciences', 'Biotech')
ON CONFLICT (ticker) DO NOTHING;

COMMENT ON TABLE insider_transactions IS 'SEC Form 4 insider trading transactions';
COMMENT ON TABLE institutional_holdings IS 'SEC 13F institutional investor holdings';
COMMENT ON TABLE job_postings IS 'Job postings from various sources';
COMMENT ON TABLE executive_changes IS 'Executive departures/hires from SEC 8-K filings';
COMMENT ON TABLE signal_scores IS 'Aggregated signal scores per company per day';
COMMENT ON TABLE signals IS 'Individual signals contributing to scores';
