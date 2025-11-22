-- Patent/IP Intelligence System - Database Schema
-- PostgreSQL 14+

-- Create database (run this separately as superuser)
-- CREATE DATABASE patent_intelligence;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- =============================================================================
-- CORE TABLES
-- =============================================================================

-- Drugs table: Core entity for branded pharmaceutical products
CREATE TABLE IF NOT EXISTS drugs (
    drug_id SERIAL PRIMARY KEY,
    nda_number VARCHAR(20) UNIQUE,  -- FDA New Drug Application number
    brand_name VARCHAR(200) NOT NULL,
    generic_name VARCHAR(200),
    active_ingredient VARCHAR(500),
    branded_company VARCHAR(200),
    branded_company_ticker VARCHAR(10),
    therapeutic_area VARCHAR(200),
    dosage_form VARCHAR(100),
    route_of_administration VARCHAR(100),
    annual_revenue BIGINT,  -- USD
    revenue_year INTEGER,
    fda_approval_date DATE,
    market_status VARCHAR(50) DEFAULT 'ACTIVE',  -- ACTIVE, DISCONTINUED
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create index for common queries
CREATE INDEX IF NOT EXISTS idx_drugs_brand_name ON drugs(brand_name);
CREATE INDEX IF NOT EXISTS idx_drugs_generic_name ON drugs(generic_name);
CREATE INDEX IF NOT EXISTS idx_drugs_company_ticker ON drugs(branded_company_ticker);
CREATE INDEX IF NOT EXISTS idx_drugs_therapeutic_area ON drugs(therapeutic_area);
CREATE INDEX IF NOT EXISTS idx_drugs_annual_revenue ON drugs(annual_revenue DESC);

-- Patents table: Patent information linked to drugs
CREATE TABLE IF NOT EXISTS patents (
    patent_id SERIAL PRIMARY KEY,
    patent_number VARCHAR(20) UNIQUE NOT NULL,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    patent_type VARCHAR(50),  -- COMPOSITION, METHOD_OF_USE, FORMULATION, PROCESS
    patent_use_code VARCHAR(50),  -- Orange Book use code
    patent_claims TEXT,
    filing_date DATE,
    grant_date DATE,
    base_expiration_date DATE NOT NULL,  -- Original 20-year expiration
    pta_days INTEGER DEFAULT 0,  -- Patent Term Adjustment (days)
    pte_days INTEGER DEFAULT 0,  -- Patent Term Extension (days)
    adjusted_expiration_date DATE,  -- After PTA + PTE
    pediatric_exclusivity_date DATE,  -- 6-month pediatric extension
    final_expiration_date DATE,  -- Latest possible expiration
    exclusivity_code VARCHAR(50),
    exclusivity_expiration DATE,
    patent_status VARCHAR(50) DEFAULT 'ACTIVE',  -- ACTIVE, EXPIRED, INVALIDATED
    strength_score INTEGER CHECK (strength_score >= 1 AND strength_score <= 10),
    data_source VARCHAR(50) DEFAULT 'ORANGE_BOOK',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for patents
CREATE INDEX IF NOT EXISTS idx_patents_drug_id ON patents(drug_id);
CREATE INDEX IF NOT EXISTS idx_patents_expiration ON patents(final_expiration_date);
CREATE INDEX IF NOT EXISTS idx_patents_status ON patents(patent_status);
CREATE INDEX IF NOT EXISTS idx_patents_type ON patents(patent_type);

-- Generic Applications table: ANDA filings and approvals
CREATE TABLE IF NOT EXISTS generic_applications (
    anda_id SERIAL PRIMARY KEY,
    anda_number VARCHAR(20) UNIQUE NOT NULL,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    generic_company VARCHAR(200) NOT NULL,
    generic_company_ticker VARCHAR(10),
    generic_drug_name VARCHAR(200),
    dosage_form VARCHAR(100),
    strength VARCHAR(100),
    filing_date DATE,
    first_to_file BOOLEAN DEFAULT FALSE,
    paragraph_iv_certification BOOLEAN DEFAULT FALSE,
    tentative_approval_date DATE,
    final_approval_date DATE,
    status VARCHAR(50) DEFAULT 'PENDING',  -- PENDING, TENTATIVE, APPROVED, WITHDRAWN
    approval_notes TEXT,
    data_source VARCHAR(50) DEFAULT 'FDA',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for generic applications
CREATE INDEX IF NOT EXISTS idx_anda_drug_id ON generic_applications(drug_id);
CREATE INDEX IF NOT EXISTS idx_anda_company ON generic_applications(generic_company);
CREATE INDEX IF NOT EXISTS idx_anda_status ON generic_applications(status);
CREATE INDEX IF NOT EXISTS idx_anda_approval_date ON generic_applications(final_approval_date);
CREATE INDEX IF NOT EXISTS idx_anda_first_to_file ON generic_applications(first_to_file);

-- Litigation table: Patent challenges and court cases
CREATE TABLE IF NOT EXISTS litigation (
    litigation_id SERIAL PRIMARY KEY,
    case_id VARCHAR(50) UNIQUE NOT NULL,
    case_name VARCHAR(500),
    patent_number VARCHAR(20) REFERENCES patents(patent_number) ON DELETE CASCADE,
    anda_number VARCHAR(20) REFERENCES generic_applications(anda_number) ON DELETE SET NULL,
    plaintiff VARCHAR(200),  -- Usually branded company
    defendant VARCHAR(200),  -- Usually generic company
    court VARCHAR(200),
    jurisdiction VARCHAR(100),
    case_type VARCHAR(50),  -- PARAGRAPH_IV, IPR, PGR
    filing_date DATE,
    trial_date DATE,
    decision_date DATE,
    outcome VARCHAR(50),  -- PATENT_UPHELD, PATENT_INVALIDATED, SETTLED, ONGOING
    settlement_terms TEXT,
    damages_awarded BIGINT,
    case_summary TEXT,
    pacer_link VARCHAR(500),
    data_source VARCHAR(50) DEFAULT 'MANUAL',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for litigation
CREATE INDEX IF NOT EXISTS idx_litigation_patent ON litigation(patent_number);
CREATE INDEX IF NOT EXISTS idx_litigation_anda ON litigation(anda_number);
CREATE INDEX IF NOT EXISTS idx_litigation_outcome ON litigation(outcome);
CREATE INDEX IF NOT EXISTS idx_litigation_decision_date ON litigation(decision_date);

-- Patent Cliff Calendar: Materialized view of upcoming events
CREATE TABLE IF NOT EXISTS patent_cliff_calendar (
    event_id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,  -- PATENT_EXPIRATION, ANDA_APPROVAL, COURT_DECISION, EXCLUSIVITY_END
    event_date DATE NOT NULL,
    related_patent_number VARCHAR(20),
    related_anda_number VARCHAR(20),
    certainty_score DECIMAL(5,2),  -- 0-100
    market_opportunity BIGINT,  -- USD at risk
    opportunity_tier VARCHAR(50),  -- BLOCKBUSTER, HIGH_VALUE, MEDIUM_VALUE, SMALL
    trade_recommendation TEXT,
    recommendation_confidence VARCHAR(20),  -- HIGH, MEDIUM, LOW
    notes TEXT,
    is_processed BOOLEAN DEFAULT FALSE,
    alert_sent BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for patent cliff calendar
CREATE INDEX IF NOT EXISTS idx_calendar_drug_id ON patent_cliff_calendar(drug_id);
CREATE INDEX IF NOT EXISTS idx_calendar_event_date ON patent_cliff_calendar(event_date);
CREATE INDEX IF NOT EXISTS idx_calendar_event_type ON patent_cliff_calendar(event_type);
CREATE INDEX IF NOT EXISTS idx_calendar_certainty ON patent_cliff_calendar(certainty_score DESC);
CREATE INDEX IF NOT EXISTS idx_calendar_opportunity ON patent_cliff_calendar(market_opportunity DESC);

-- =============================================================================
-- SUPPORTING TABLES
-- =============================================================================

-- Drug Revenue History: Track annual revenues over time
CREATE TABLE IF NOT EXISTS drug_revenue_history (
    revenue_id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    fiscal_year INTEGER NOT NULL,
    annual_revenue BIGINT,
    revenue_source VARCHAR(100),  -- 10-K, IQVIA, ESTIMATE
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(drug_id, fiscal_year)
);

CREATE INDEX IF NOT EXISTS idx_revenue_drug_id ON drug_revenue_history(drug_id);
CREATE INDEX IF NOT EXISTS idx_revenue_year ON drug_revenue_history(fiscal_year DESC);

-- Company mapping: Track pharmaceutical companies
CREATE TABLE IF NOT EXISTS companies (
    company_id SERIAL PRIMARY KEY,
    company_name VARCHAR(200) NOT NULL,
    ticker_symbol VARCHAR(10),
    company_type VARCHAR(50),  -- BRANDED, GENERIC, BOTH
    headquarters_country VARCHAR(100),
    market_cap BIGINT,
    is_public BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_name)
);

CREATE INDEX IF NOT EXISTS idx_companies_ticker ON companies(ticker_symbol);
CREATE INDEX IF NOT EXISTS idx_companies_type ON companies(company_type);

-- Exclusivity Periods: Track various exclusivity types
CREATE TABLE IF NOT EXISTS exclusivity_periods (
    exclusivity_id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES drugs(drug_id) ON DELETE CASCADE,
    exclusivity_type VARCHAR(50) NOT NULL,  -- NCE, ORPHAN, PEDIATRIC, NEW_INDICATION
    start_date DATE,
    end_date DATE NOT NULL,
    notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_exclusivity_drug_id ON exclusivity_periods(drug_id);
CREATE INDEX IF NOT EXISTS idx_exclusivity_end_date ON exclusivity_periods(end_date);

-- ETL Job Tracking: Monitor data pipeline runs
CREATE TABLE IF NOT EXISTS etl_jobs (
    job_id SERIAL PRIMARY KEY,
    job_name VARCHAR(100) NOT NULL,
    job_type VARCHAR(50),  -- EXTRACTION, TRANSFORMATION, LOADING
    data_source VARCHAR(50),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'RUNNING',  -- RUNNING, SUCCESS, FAILED
    records_processed INTEGER DEFAULT 0,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_etl_jobs_name ON etl_jobs(job_name);
CREATE INDEX IF NOT EXISTS idx_etl_jobs_status ON etl_jobs(status);
CREATE INDEX IF NOT EXISTS idx_etl_jobs_start_time ON etl_jobs(start_time DESC);

-- Alert History: Track sent notifications
CREATE TABLE IF NOT EXISTS alert_history (
    alert_id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES patent_cliff_calendar(event_id) ON DELETE SET NULL,
    alert_type VARCHAR(50) NOT NULL,  -- EMAIL, SLACK, SMS
    recipient VARCHAR(200),
    subject VARCHAR(500),
    body TEXT,
    sent_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'PENDING',  -- PENDING, SENT, FAILED
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_alerts_event_id ON alert_history(event_id);
CREATE INDEX IF NOT EXISTS idx_alerts_sent_at ON alert_history(sent_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alert_history(status);

-- =============================================================================
-- VIEWS
-- =============================================================================

-- View: Upcoming Patent Expirations (next 18 months)
CREATE OR REPLACE VIEW v_upcoming_patent_expirations AS
SELECT
    d.drug_id,
    d.brand_name,
    d.generic_name,
    d.branded_company,
    d.branded_company_ticker,
    d.annual_revenue,
    p.patent_number,
    p.patent_type,
    p.final_expiration_date,
    p.patent_status,
    EXTRACT(DAY FROM (p.final_expiration_date - CURRENT_DATE)) AS days_until_expiration,
    CASE
        WHEN d.annual_revenue >= 1000000000 THEN 'BLOCKBUSTER'
        WHEN d.annual_revenue >= 500000000 THEN 'HIGH_VALUE'
        WHEN d.annual_revenue >= 100000000 THEN 'MEDIUM_VALUE'
        ELSE 'SMALL'
    END AS opportunity_tier
FROM drugs d
JOIN patents p ON d.drug_id = p.drug_id
WHERE p.final_expiration_date BETWEEN CURRENT_DATE AND (CURRENT_DATE + INTERVAL '18 months')
  AND p.patent_status = 'ACTIVE'
ORDER BY p.final_expiration_date ASC;

-- View: Generic Competition Summary
CREATE OR REPLACE VIEW v_generic_competition_summary AS
SELECT
    d.drug_id,
    d.brand_name,
    d.generic_name,
    d.annual_revenue,
    COUNT(DISTINCT ga.anda_number) AS total_anda_filings,
    COUNT(DISTINCT CASE WHEN ga.status = 'APPROVED' THEN ga.anda_number END) AS approved_andas,
    COUNT(DISTINCT CASE WHEN ga.first_to_file THEN ga.anda_number END) AS first_to_file_count,
    MIN(ga.final_approval_date) AS first_generic_approval_date,
    ARRAY_AGG(DISTINCT ga.generic_company) FILTER (WHERE ga.status = 'APPROVED') AS approved_generic_companies
FROM drugs d
LEFT JOIN generic_applications ga ON d.drug_id = ga.drug_id
GROUP BY d.drug_id, d.brand_name, d.generic_name, d.annual_revenue
ORDER BY d.annual_revenue DESC NULLS LAST;

-- View: Active Litigation Summary
CREATE OR REPLACE VIEW v_active_litigation AS
SELECT
    d.brand_name,
    d.generic_name,
    l.case_id,
    l.case_name,
    l.plaintiff,
    l.defendant,
    l.court,
    l.case_type,
    l.filing_date,
    l.outcome,
    p.patent_number,
    p.final_expiration_date
FROM litigation l
JOIN patents p ON l.patent_number = p.patent_number
JOIN drugs d ON p.drug_id = d.drug_id
WHERE l.outcome IS NULL OR l.outcome = 'ONGOING'
ORDER BY l.filing_date DESC;

-- View: Patent Cliff Calendar (12-month view)
CREATE OR REPLACE VIEW v_patent_cliff_calendar_12m AS
SELECT
    pcc.event_id,
    d.brand_name,
    d.generic_name,
    d.branded_company,
    d.branded_company_ticker,
    pcc.event_type,
    pcc.event_date,
    pcc.certainty_score,
    pcc.market_opportunity,
    pcc.opportunity_tier,
    pcc.trade_recommendation,
    pcc.recommendation_confidence,
    EXTRACT(DAY FROM (pcc.event_date - CURRENT_DATE)) AS days_until_event
FROM patent_cliff_calendar pcc
JOIN drugs d ON pcc.drug_id = d.drug_id
WHERE pcc.event_date BETWEEN CURRENT_DATE AND (CURRENT_DATE + INTERVAL '12 months')
ORDER BY pcc.event_date ASC, pcc.certainty_score DESC;

-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function: Calculate adjusted patent expiration date
CREATE OR REPLACE FUNCTION calculate_adjusted_expiration(
    p_base_expiration DATE,
    p_pta_days INTEGER,
    p_pte_days INTEGER,
    p_pediatric_extension BOOLEAN DEFAULT FALSE
) RETURNS DATE AS $$
BEGIN
    RETURN p_base_expiration
           + (COALESCE(p_pta_days, 0) || ' days')::INTERVAL
           + (COALESCE(p_pte_days, 0) || ' days')::INTERVAL
           + (CASE WHEN p_pediatric_extension THEN INTERVAL '6 months' ELSE INTERVAL '0' END);
END;
$$ LANGUAGE plpgsql;

-- Function: Calculate patent cliff certainty score
CREATE OR REPLACE FUNCTION calculate_certainty_score(
    p_drug_id INTEGER
) RETURNS DECIMAL(5,2) AS $$
DECLARE
    v_patent_expired_score DECIMAL(5,2);
    v_no_litigation_score DECIMAL(5,2);
    v_anda_approved_score DECIMAL(5,2);
    v_no_extension_score DECIMAL(5,2);
    v_total_score DECIMAL(5,2);
    v_expired_patents INTEGER;
    v_total_patents INTEGER;
    v_active_litigation INTEGER;
    v_approved_andas INTEGER;
BEGIN
    -- Check patent expiration status
    SELECT
        COUNT(*) FILTER (WHERE final_expiration_date <= CURRENT_DATE),
        COUNT(*)
    INTO v_expired_patents, v_total_patents
    FROM patents
    WHERE drug_id = p_drug_id AND patent_status = 'ACTIVE';

    IF v_total_patents > 0 THEN
        v_patent_expired_score := (v_expired_patents::DECIMAL / v_total_patents) * 100;
    ELSE
        v_patent_expired_score := 100;
    END IF;

    -- Check litigation status
    SELECT COUNT(*)
    INTO v_active_litigation
    FROM litigation l
    JOIN patents p ON l.patent_number = p.patent_number
    WHERE p.drug_id = p_drug_id AND (l.outcome IS NULL OR l.outcome = 'ONGOING');

    IF v_active_litigation = 0 THEN
        v_no_litigation_score := 100;
    ELSE
        v_no_litigation_score := 0;
    END IF;

    -- Check ANDA approvals
    SELECT COUNT(*)
    INTO v_approved_andas
    FROM generic_applications
    WHERE drug_id = p_drug_id AND status = 'APPROVED';

    IF v_approved_andas >= 3 THEN
        v_anda_approved_score := 100;
    ELSIF v_approved_andas >= 1 THEN
        v_anda_approved_score := 50;
    ELSE
        v_anda_approved_score := 0;
    END IF;

    -- Default no extension score (simplified)
    v_no_extension_score := 80;

    -- Calculate weighted total
    v_total_score := (
        (0.40 * v_patent_expired_score) +
        (0.30 * v_no_litigation_score) +
        (0.20 * v_anda_approved_score) +
        (0.10 * v_no_extension_score)
    );

    RETURN ROUND(v_total_score, 2);
END;
$$ LANGUAGE plpgsql;

-- Function: Update timestamps automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- TRIGGERS
-- =============================================================================

-- Auto-update timestamps
CREATE TRIGGER update_drugs_updated_at
    BEFORE UPDATE ON drugs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patents_updated_at
    BEFORE UPDATE ON patents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_generic_applications_updated_at
    BEFORE UPDATE ON generic_applications
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_litigation_updated_at
    BEFORE UPDATE ON litigation
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patent_cliff_calendar_updated_at
    BEFORE UPDATE ON patent_cliff_calendar
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_companies_updated_at
    BEFORE UPDATE ON companies
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- SAMPLE DATA INSERT (Top Drugs for Testing)
-- =============================================================================

-- Note: This sample data is for testing purposes
-- Real data should be loaded via the ETL pipeline

INSERT INTO companies (company_name, ticker_symbol, company_type, headquarters_country, is_public)
VALUES
    ('AbbVie Inc.', 'ABBV', 'BRANDED', 'USA', TRUE),
    ('Bristol-Myers Squibb', 'BMY', 'BRANDED', 'USA', TRUE),
    ('Merck & Co.', 'MRK', 'BRANDED', 'USA', TRUE),
    ('Pfizer Inc.', 'PFE', 'BOTH', 'USA', TRUE),
    ('Johnson & Johnson', 'JNJ', 'BOTH', 'USA', TRUE),
    ('Eli Lilly', 'LLY', 'BRANDED', 'USA', TRUE),
    ('Novartis', 'NVS', 'BOTH', 'Switzerland', TRUE),
    ('Teva Pharmaceutical', 'TEVA', 'GENERIC', 'Israel', TRUE),
    ('Viatris', 'VTRS', 'GENERIC', 'USA', TRUE),
    ('Sandoz', NULL, 'GENERIC', 'Switzerland', FALSE)
ON CONFLICT (company_name) DO NOTHING;

-- Grant permissions (adjust as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO patent_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO patent_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO patent_user;

-- =============================================================================
-- END OF SCHEMA
-- =============================================================================
