# Patent/IP Intelligence System - Technical Specification

## Overview
Automated system for tracking pharmaceutical patent expirations, generic drug approvals, and patent litigation to identify investment opportunities in both branded and generic pharmaceutical companies.

## Business Objective
- **Primary Goal**: Identify patent cliff events 6-18 months in advance
- **Success Metric**: Predict generic competition entry dates with 90%+ accuracy
- **Target ROI**: 20-40% returns on pair trades (long generic, short branded)

---

## Data Sources

### 1. FDA Orange Book
- **Endpoint**: https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files
- **Update Frequency**: Monthly (official updates)
- **Key Data**:
  - Patent numbers for each branded drug
  - Patent expiration dates
  - Exclusivity periods (pediatric, orphan, etc.)
  - Generic approvals (ANDA numbers)

### 2. USPTO Patent Database
- **Endpoint**: https://patentsview.org/api/
- **Alternative**: https://developer.uspto.gov/
- **Update Frequency**: Weekly
- **Key Data**:
  - Full patent text and claims
  - Continuation applications
  - Patent term extensions (PTE)
  - Patent family relationships

### 3. PACER (Court Filings)
- **Endpoint**: https://pacer.uscourts.gov/
- **Update Frequency**: Real-time
- **Key Data**:
  - Hatch-Waxman litigation (Paragraph IV challenges)
  - ANDA litigation outcomes
  - Settlement agreements
  - IPR (Inter Partes Review) challenges at PTAB

### 4. FDA ANDA Pipeline
- **Endpoint**: https://www.fda.gov/drugs/abbreviated-new-drug-application-anda
- **Update Frequency**: Monthly
- **Key Data**:
  - Generic applications filed (company + drug)
  - First-to-file status (180-day exclusivity)
  - Tentative approvals
  - Approval actions

### 5. SEC Filings (Generic Companies)
- **Endpoint**: SEC EDGAR API
- **Update Frequency**: Real-time
- **Key Data**:
  - Generic pipeline disclosures
  - Litigation updates
  - Product launch expectations

### 6. Drug Sales Data
- **Sources**: Company 10-Ks, IQVIA (if budget allows)
- **Update Frequency**: Quarterly
- **Key Data**:
  - Revenue by product
  - Market share
  - Prescription volume trends

---

## Data Pipeline Architecture

### Stack
- **Language**: Python 3.11+
- **Data Storage**: PostgreSQL (relational data) + Elasticsearch (patent text search)
- **ETL**: Apache Airflow for orchestration
- **Cloud**: AWS (EC2 for processing, RDS, S3)
- **AI**: Claude/GPT-4 for patent claim analysis and litigation outcome prediction

### Pipeline Flow
```
┌──────────────────────────────────────────────────────────────┐
│                      Data Ingestion                          │
├──────────────────────────────────────────────────────────────┤
│   Orange Book    USPTO    PACER    ANDA    SEC    Sales     │
│       ↓           ↓        ↓        ↓       ↓       ↓        │
│   [Monthly]   [Weekly]  [Daily]  [Monthly] [RT]  [Quarterly]│
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                       │
├──────────────────────────────────────────────────────────────┤
│  • Patent expiration calculation (including extensions)       │
│  • Litigation status tracking (active vs settled)             │
│  • Generic competition timeline modeling                      │
│  • Market size estimation (revenue at risk)                   │
│  • AI patent claim analysis (strength assessment)             │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                    Opportunity Scoring                        │
├──────────────────────────────────────────────────────────────┤
│  • Patent cliff certainty (0-100%)                            │
│  • Market opportunity size ($M)                               │
│  • Timeline to generic entry (months)                         │
│  • Number of generic filers (competition intensity)           │
│  • Trade recommendation (long/short/pair)                     │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│                      Output & Alerts                          │
├──────────────────────────────────────────────────────────────┤
│  • Weekly patent cliff calendar (12-month forward view)       │
│  • Litigation alert emails (new filings, decisions)           │
│  • Generic approval alerts (FDA actions)                      │
│  • Dashboard with trade ideas                                 │
└──────────────────────────────────────────────────────────────┘
```

---

## Investment Logic

### Patent Cliff Scenarios

#### Scenario 1: Certain Generic Entry
**Setup**: All patents expired or invalidated, multiple ANDAs approved
**Trade**:
- Short branded drug company 6-12 months before generic launch
- Long generic manufacturer with first-to-file status
- Target: 30-50% move in branded stock (down), 20-40% in generic (up)

#### Scenario 2: Uncertain Litigation
**Setup**: Active Paragraph IV challenge, unclear outcome
**Trade**:
- Wait for court decision
- If patents invalidated: Execute Scenario 1
- If patents upheld: Long branded drug company (bounce on relief)

#### Scenario 3: Patent Extension Granted
**Setup**: Company wins Patent Term Extension (PTE)
**Trade**:
- Long branded drug company (unexpected exclusivity extension)
- Short any generic companies that were positioning for launch

---

## Scoring Model

### Patent Cliff Certainty Score (0-100%)

| Factor | Weight | Scoring Logic |
|--------|--------|---------------|
| Primary patent expired | 40% | 0% if still active, 100% if expired |
| No active litigation | 30% | 0% if litigating, 100% if clear |
| ANDA approvals granted | 20% | 0% if none, 100% if 3+ approved |
| Historical PTE unlikely | 10% | Based on drug class, indication |

**Example Calculation**:
```python
certainty_score = (
    (0.4 * patent_expired) +
    (0.3 * no_litigation) +
    (0.2 * anda_approved) +
    (0.1 * no_extension_expected)
) * 100

if certainty_score >= 80:
    trade_recommendation = "HIGH CONFIDENCE - Execute trade"
elif certainty_score >= 60:
    trade_recommendation = "MEDIUM - Monitor closely"
else:
    trade_recommendation = "LOW - Too uncertain"
```

### Market Opportunity Size

```python
annual_revenue = get_product_sales(drug_name, year)
generic_erosion_rate = 0.80  # Branded drug typically loses 80% revenue
market_opportunity = annual_revenue * generic_erosion_rate

if market_opportunity > 1_000_000_000:  # $1B+
    opportunity_tier = "BLOCKBUSTER"
elif market_opportunity > 500_000_000:
    opportunity_tier = "HIGH VALUE"
elif market_opportunity > 100_000_000:
    opportunity_tier = "MEDIUM VALUE"
else:
    opportunity_tier = "SMALL - Skip"
```

---

## Database Schema

### drugs
```sql
CREATE TABLE drugs (
    drug_id SERIAL PRIMARY KEY,
    brand_name VARCHAR(200),
    generic_name VARCHAR(200),
    branded_company_ticker VARCHAR(10),
    therapeutic_area VARCHAR(200),
    annual_revenue BIGINT,  -- USD
    fda_approval_date DATE
);
```

### patents
```sql
CREATE TABLE patents (
    patent_number VARCHAR(20) PRIMARY KEY,
    drug_id INT REFERENCES drugs(drug_id),
    patent_type VARCHAR(50),  -- Composition, Method of Use, Formulation
    filing_date DATE,
    grant_date DATE,
    expiration_date DATE,
    adjusted_expiration DATE,  -- After PTE
    patent_claims TEXT,
    strength_score INT  -- 1-10, AI-assessed
);
```

### generic_applications
```sql
CREATE TABLE generic_applications (
    anda_number VARCHAR(20) PRIMARY KEY,
    drug_id INT REFERENCES drugs(drug_id),
    generic_company VARCHAR(200),
    generic_company_ticker VARCHAR(10),
    filing_date DATE,
    first_to_file BOOLEAN,
    approval_date DATE,
    tentative_approval_date DATE,
    status VARCHAR(50)
);
```

### litigation
```sql
CREATE TABLE litigation (
    case_id VARCHAR(50) PRIMARY KEY,
    patent_number VARCHAR(20) REFERENCES patents(patent_number),
    anda_number VARCHAR(20) REFERENCES generic_applications(anda_number),
    court VARCHAR(100),
    filing_date DATE,
    decision_date DATE,
    outcome VARCHAR(50),  -- Patent upheld, Invalidated, Settled
    settlement_terms TEXT
);
```

### patent_cliff_calendar
```sql
CREATE TABLE patent_cliff_calendar (
    event_id SERIAL PRIMARY KEY,
    drug_id INT REFERENCES drugs(drug_id),
    event_type VARCHAR(50),  -- Patent expiration, ANDA approval, Court decision
    event_date DATE,
    certainty_score DECIMAL(5,2),
    market_opportunity BIGINT,
    trade_recommendation TEXT
);
```

---

## AI-Powered Features

### 1. Patent Claim Strength Analysis
Use Claude/GPT-4 to analyze patent claims and predict litigation outcomes:

```python
prompt = f"""
Analyze this pharmaceutical patent claim:

Patent Number: {patent_number}
Claims: {patent_claims}
Drug: {drug_name}

Assess:
1. Claim breadth (narrow/moderate/broad)
2. Likelihood of surviving Paragraph IV challenge (0-100%)
3. Key vulnerabilities
4. Comparison to similar litigated patents

Provide score and 2-sentence reasoning.
"""
```

### 2. Litigation Outcome Prediction
Train model on historical Hatch-Waxman cases:

```python
features = [
    'patent_age',
    'claim_count',
    'prior_invalidations_same_company',
    'court_jurisdiction',
    'generic_challenger_success_rate'
]

model = RandomForestClassifier()
# Trained on 500+ historical cases
predicted_outcome = model.predict(features)
```

---

## MVP Scope (Week 1-2)

### Must Have
- [x] Orange Book data ingestion (top 50 drugs by revenue)
- [x] USPTO patent expiration calculation
- [x] ANDA approval tracking
- [x] Basic patent cliff calendar (12-month view)
- [x] Weekly email digest of upcoming events

### Nice to Have (Future)
- [ ] PACER litigation tracking
- [ ] AI patent claim analysis
- [ ] Historical litigation outcome model
- [ ] Web dashboard with trade ideas
- [ ] Real-time alerts for FDA approvals

---

## Initial Target List

### Top 20 Drugs Facing Patent Cliffs (2024-2026)

Focus on:
1. Blockbusters (>$1B annual revenue)
2. Patents expiring 2024-2026
3. Multiple ANDA filers (high certainty)
4. Public companies (tradeable)

**Examples**:
- Humira (AbbVie) - Already faced cliff in 2023, study for model training
- Eliquis (Bristol Myers Squibb) - Expires 2026
- Keytruda (Merck) - Expires 2028
- Jardiance (Boehringer/Lilly) - Expires 2025

---

## Risk Management

### False Signals
- Patent extension granted unexpectedly
- **Mitigation**: Track PTE application history, only trade high-certainty (>80%) events

### Litigation Surprises
- Unexpected settlement or court decision
- **Mitigation**: Daily PACER monitoring, position sizing (max 5% per trade)

### Generic Delay Tactics
- Authorized generics, rebate wars
- **Mitigation**: Monitor branded company conference calls for defensive strategies

---

## Success Metrics

### System Performance
- **Patent expiration accuracy**: 95%+ (straightforward calculation)
- **ANDA approval detection latency**: < 24 hours
- **Litigation outcome prediction**: 70%+ accuracy (AI model)

### Investment Performance
- **Win rate**: 65%+ of trades profitable
- **Average return**: 25%+ per pair trade
- **Max drawdown**: < 15%

---

## Budget

| Item | Cost |
|------|------|
| PACER account | $30/month |
| AWS infrastructure | $100/month |
| AI API (patent analysis) | $150/month |
| Database hosting | $50/month |
| **Total** | **$330/month** |

**One-time setup**: ~$400

---

## Timeline

- **Week 1**: Orange Book + USPTO data pipelines
- **Week 2**: Patent cliff calendar + scoring model
- **Week 3**: ANDA tracking + email alerts
- **Week 4**: Backtesting on 2023 patent cliffs
- **Week 5+**: Live monitoring + litigation tracking

---

## Next Steps

1. Download Orange Book data files
2. Build USPTO API scraper
3. Create patent expiration calculator
4. Set up PostgreSQL database
5. Implement scoring model
6. Generate first patent cliff calendar
