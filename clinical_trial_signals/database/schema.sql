-- Clinical Trial Signal Detection System - Database Schema
-- PostgreSQL 14+

-- Companies table (must be created first due to foreign key reference)
CREATE TABLE IF NOT EXISTS companies (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL,
    market_cap BIGINT,
    current_price DECIMAL(10,2),
    sector VARCHAR(100) DEFAULT 'Biotechnology',
    cik VARCHAR(20),  -- SEC Central Index Key
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trials table
CREATE TABLE IF NOT EXISTS trials (
    trial_id VARCHAR(20) PRIMARY KEY,  -- NCT number
    company_ticker VARCHAR(10) REFERENCES companies(ticker),
    drug_name VARCHAR(200),
    indication VARCHAR(500),
    phase VARCHAR(10),
    enrollment_target INT,
    enrollment_current INT,
    start_date DATE,
    expected_completion DATE,
    primary_completion_date DATE,
    primary_endpoint TEXT,
    status VARCHAR(50),
    sponsor VARCHAR(300),
    study_type VARCHAR(50),
    last_updated TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_data JSONB  -- Store full API response for change detection
);

-- Trial signals table
CREATE TABLE IF NOT EXISTS trial_signals (
    signal_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(20) REFERENCES trials(trial_id),
    signal_type VARCHAR(100) NOT NULL,
    signal_value TEXT,
    signal_weight INT CHECK (signal_weight >= -5 AND signal_weight <= 5),
    detected_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100),
    source_url TEXT,
    raw_data JSONB,
    processed BOOLEAN DEFAULT FALSE
);

-- Trial scores table
CREATE TABLE IF NOT EXISTS trial_scores (
    score_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(20) REFERENCES trials(trial_id),
    score_date DATE DEFAULT CURRENT_DATE,
    composite_score DECIMAL(4,2) CHECK (composite_score >= 0 AND composite_score <= 10),
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    recommendation VARCHAR(20),
    contributing_signals JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(trial_id, score_date)
);

-- SEC filings table for tracking 8-K and other filings
CREATE TABLE IF NOT EXISTS sec_filings (
    filing_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10) REFERENCES companies(ticker),
    filing_type VARCHAR(20),  -- 8-K, 10-Q, 10-K
    filing_date DATE,
    accession_number VARCHAR(30) UNIQUE,
    filing_url TEXT,
    description TEXT,
    raw_content TEXT,
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Trial history for change detection
CREATE TABLE IF NOT EXISTS trial_history (
    history_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(20) REFERENCES trials(trial_id),
    snapshot_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    enrollment_current INT,
    status VARCHAR(50),
    expected_completion DATE,
    primary_endpoint TEXT,
    raw_data JSONB
);

-- Email digest logs
CREATE TABLE IF NOT EXISTS email_digests (
    digest_id SERIAL PRIMARY KEY,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    recipients TEXT[],
    subject VARCHAR(300),
    signals_count INT,
    content_hash VARCHAR(64)  -- To prevent duplicate sends
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_trials_ticker ON trials(company_ticker);
CREATE INDEX IF NOT EXISTS idx_trials_status ON trials(status);
CREATE INDEX IF NOT EXISTS idx_trials_phase ON trials(phase);
CREATE INDEX IF NOT EXISTS idx_signals_trial ON trial_signals(trial_id);
CREATE INDEX IF NOT EXISTS idx_signals_date ON trial_signals(detected_date);
CREATE INDEX IF NOT EXISTS idx_signals_type ON trial_signals(signal_type);
CREATE INDEX IF NOT EXISTS idx_scores_trial ON trial_scores(trial_id);
CREATE INDEX IF NOT EXISTS idx_scores_date ON trial_scores(score_date);
CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings(company_ticker);
CREATE INDEX IF NOT EXISTS idx_sec_filings_date ON sec_filings(filing_date);
CREATE INDEX IF NOT EXISTS idx_trial_history_trial ON trial_history(trial_id);
CREATE INDEX IF NOT EXISTS idx_trial_history_date ON trial_history(snapshot_date);

-- Function to update timestamp on row update
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at columns
DROP TRIGGER IF EXISTS update_companies_updated_at ON companies;
CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_trials_updated_at ON trials;
CREATE TRIGGER update_trials_updated_at
    BEFORE UPDATE ON trials
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Sample data: Initial watchlist companies (biotech)
INSERT INTO companies (ticker, company_name, market_cap, sector, cik) VALUES
    ('SAVA', 'Cassava Sciences Inc', 800000000, 'Biotechnology', '1069974'),
    ('MRNA', 'Moderna Inc', 15000000000, 'Biotechnology', '1682852'),
    ('NVAX', 'Novavax Inc', 1500000000, 'Biotechnology', '1000694'),
    ('IONS', 'Ionis Pharmaceuticals', 7000000000, 'Biotechnology', '874015'),
    ('ALNY', 'Alnylam Pharmaceuticals', 25000000000, 'Biotechnology', '1178670'),
    ('BMRN', 'BioMarin Pharmaceutical', 14000000000, 'Biotechnology', '1048477'),
    ('SRPT', 'Sarepta Therapeutics', 10000000000, 'Biotechnology', '873303'),
    ('RARE', 'Ultragenyx Pharmaceutical', 5000000000, 'Biotechnology', '1564408'),
    ('NBIX', 'Neurocrine Biosciences', 13000000000, 'Biotechnology', '914475'),
    ('ACAD', 'ACADIA Pharmaceuticals', 4000000000, 'Biotechnology', '1070154')
ON CONFLICT (ticker) DO NOTHING;

-- View for active signals summary
CREATE OR REPLACE VIEW active_signals_summary AS
SELECT
    t.trial_id,
    t.drug_name,
    c.ticker,
    c.company_name,
    COUNT(ts.signal_id) as signal_count,
    SUM(ts.signal_weight) as total_weight,
    MAX(ts.detected_date) as latest_signal
FROM trials t
JOIN companies c ON t.company_ticker = c.ticker
LEFT JOIN trial_signals ts ON t.trial_id = ts.trial_id
WHERE ts.detected_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY t.trial_id, t.drug_name, c.ticker, c.company_name
ORDER BY total_weight DESC;

-- View for latest scores
CREATE OR REPLACE VIEW latest_trial_scores AS
SELECT DISTINCT ON (trial_id)
    ts.*,
    t.drug_name,
    t.indication,
    c.ticker,
    c.company_name
FROM trial_scores ts
JOIN trials t ON ts.trial_id = t.trial_id
JOIN companies c ON t.company_ticker = c.ticker
ORDER BY trial_id, score_date DESC;
