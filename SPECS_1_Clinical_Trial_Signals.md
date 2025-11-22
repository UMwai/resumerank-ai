# Clinical Trial Signal Detection System - Technical Specification

## Overview
Automated intelligence system for detecting early signals of clinical trial outcomes before public announcements. Provides 2-4 week advance warning on trial results for investment edge.

## Business Objective
- **Primary Goal**: Generate actionable investment signals on biotech stocks before trial readouts
- **Success Metric**: 70%+ accuracy in predicting trial outcome direction (positive/negative)
- **Target ROI**: 5-10x on event-driven options plays

---

## Data Sources

### 1. ClinicalTrials.gov
- **Endpoint**: https://clinicaltrials.gov/api/
- **Update Frequency**: Daily
- **Key Signals**:
  - Protocol amendments (especially endpoint changes)
  - Enrollment status changes (completed early = good/bad depending on context)
  - Study completion date changes
  - New trial sites added (enrollment going well)
  - Primary outcome measure modifications

### 2. SEC EDGAR Filings
- **Endpoint**: https://www.sec.gov/cgi-bin/browse-edgar
- **Update Frequency**: Real-time (filings within minutes)
- **Key Signals**:
  - 10-Q mentions of trial milestones
  - 8-K filings (material events)
  - Management discussion of trial progress
  - Risk factor changes related to trials

### 3. USPTO Patent Applications
- **Endpoint**: https://patentsview.org/api/
- **Update Frequency**: Weekly
- **Key Signals**:
  - New patent applications filed (often right after positive data)
  - Method of use patents (suggests efficacy)
  - Formulation patents (suggests commercial readiness)

### 4. PubMed/medRxiv Preprints
- **Endpoint**: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
- **Update Frequency**: Daily
- **Key Signals**:
  - Investigator publications before conference presentations
  - Author affiliations matching trial sites
  - Biomarker studies using trial samples

### 5. Conference Abstract Deadlines
- **Sources**: ASCO, ASH, ESMO, AAD calendars
- **Update Frequency**: Manual tracking (quarterly)
- **Key Signals**:
  - Late-breaking abstract submissions (positive data)
  - Oral presentation vs poster (hierarchy of importance)
  - Company-sponsored symposiums

### 6. FDA Calendar
- **Endpoint**: https://www.fda.gov/
- **Update Frequency**: Weekly
- **Key Signals**:
  - PDUFA (Prescription Drug User Fee Act) dates
  - Advisory Committee meeting schedules
  - Breakthrough Therapy designations
  - Fast Track designations

---

## Data Pipeline Architecture

### Stack
- **Language**: Python 3.11+
- **Data Storage**: PostgreSQL (structured data) + MongoDB (documents)
- **ETL**: Apache Airflow or Prefect for orchestration
- **Cloud**: AWS (Lambda for scrapers, RDS for database, S3 for document storage)
- **AI**: OpenAI API (GPT-4) + Anthropic Claude for document analysis

### Pipeline Flow
```
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
├─────────────────────────────────────────────────────────────┤
│  ClinicalTrials.gov    SEC EDGAR    USPTO    PubMed   FDA   │
│         ↓                  ↓          ↓        ↓       ↓    │
│  [Daily Scraper]   [Real-time]  [Weekly]  [Daily]  [Weekly] │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Processing Layer                      │
├─────────────────────────────────────────────────────────────┤
│  • Entity Resolution (match companies across sources)        │
│  • Change Detection (diff previous vs current state)         │
│  • AI Document Parsing (extract key facts from filings)      │
│  • Sentiment Analysis (is this change positive/negative?)    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Signal Scoring                          │
├─────────────────────────────────────────────────────────────┤
│  • Multi-factor model (weights different signals)            │
│  • Historical backtesting (did signals predict outcomes?)    │
│  • Confidence scoring (1-10 scale)                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Output & Alerts                          │
├─────────────────────────────────────────────────────────────┤
│  • Daily email digest (top 5 opportunities)                  │
│  • Real-time Slack alerts (high-confidence signals)          │
│  • Web dashboard (all tracked trials)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Signal Scoring Model

### Positive Signals (Trial likely to succeed)
| Signal | Weight | Evidence |
|--------|--------|----------|
| Protocol amendment adding sites | +3 | Enrollment ahead of schedule |
| Insider buying in last 30 days | +4 | Management has inside info |
| Early enrollment completion | +3 | Efficacy clear early |
| New patent application filed | +5 | Protecting IP = positive data |
| Late-breaking abstract accepted | +5 | Conference wants to feature data |
| CEO presents at investor conf | +2 | Confidence in upcoming data |

### Negative Signals (Trial likely to fail)
| Signal | Weight | Evidence |
|--------|--------|----------|
| Enrollment timeline extended | -3 | Struggling to recruit |
| Protocol amendment changing endpoints | -5 | Original endpoint not working |
| Insider selling | -4 | Executives know bad news coming |
| Trial sites removed | -4 | Operational issues |
| No conference presentations planned | -2 | Nothing positive to share |
| Increased risk factor language in 10-Q | -3 | Management hedging expectations |

### Composite Score Calculation
```python
score = sum(signal_weights) / max_possible_score * 10
confidence = (number_of_signals / 6) * historical_accuracy

if score >= 7 and confidence >= 0.7:
    recommendation = "STRONG BUY"
elif score >= 5:
    recommendation = "BUY"
elif score <= 3 and confidence >= 0.7:
    recommendation = "SHORT"
else:
    recommendation = "HOLD/MONITOR"
```

---

## Database Schema

### trials
```sql
CREATE TABLE trials (
    trial_id VARCHAR(20) PRIMARY KEY,  -- NCT number
    company_ticker VARCHAR(10),
    drug_name VARCHAR(200),
    indication VARCHAR(500),
    phase VARCHAR(10),
    enrollment_target INT,
    enrollment_current INT,
    start_date DATE,
    expected_completion DATE,
    primary_endpoint TEXT,
    status VARCHAR(50),
    last_updated TIMESTAMP
);
```

### trial_signals
```sql
CREATE TABLE trial_signals (
    signal_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(20) REFERENCES trials(trial_id),
    signal_type VARCHAR(100),
    signal_value TEXT,
    signal_weight INT,  -- -5 to +5
    detected_date TIMESTAMP,
    source VARCHAR(100),
    raw_data JSONB
);
```

### trial_scores
```sql
CREATE TABLE trial_scores (
    score_id SERIAL PRIMARY KEY,
    trial_id VARCHAR(20) REFERENCES trials(trial_id),
    score_date DATE,
    composite_score DECIMAL(3,2),  -- 0 to 10
    confidence DECIMAL(3,2),  -- 0 to 1
    recommendation VARCHAR(20),
    contributing_signals JSONB
);
```

### companies
```sql
CREATE TABLE companies (
    ticker VARCHAR(10) PRIMARY KEY,
    company_name VARCHAR(200),
    market_cap BIGINT,
    current_price DECIMAL(10,2),
    sector VARCHAR(100)
);
```

---

## API Endpoints (Internal Use)

### GET /api/v1/trials/monitored
Returns list of all trials currently being tracked

### GET /api/v1/trials/{trial_id}/signals
Returns all signals detected for a specific trial

### GET /api/v1/trials/opportunities
Returns trials scored >= 7 or <= 3 (actionable opportunities)

### POST /api/v1/trials/track
Add a new trial to monitoring system

### GET /api/v1/dashboard/summary
Returns daily summary for email digest

---

## MVP Scope (Week 1-2)

### Must Have
- [x] ClinicalTrials.gov scraper for top 20 Phase 3 trials
- [x] SEC EDGAR scraper for 8-K filings from biotech companies
- [x] Basic change detection (protocol amendments, enrollment changes)
- [x] PostgreSQL database with core tables
- [x] Simple scoring model (3-5 signals)
- [x] Daily email digest of changes

### Nice to Have (Future)
- [ ] USPTO patent tracking
- [ ] PubMed preprint monitoring
- [ ] Conference abstract tracking
- [ ] Web dashboard
- [ ] Slack integration
- [ ] Historical backtesting framework

---

## Configuration

### Monitored Trial Criteria
- Phase 2/3 or Phase 3 only
- Public company (has ticker)
- Market cap < $5B (more volatile = bigger moves)
- Binary event (clear readout date)
- High unmet need indication (higher probability of success)

### Initial Watchlist (Top 20 Trials)
Will be populated based on:
1. Upcoming PDUFA dates in next 6 months
2. Phase 3 trials in oncology, rare disease, neurology
3. Companies with market cap $500M - $3B (sweet spot for volatility)

---

## Risk Management

### False Positives
- Signal appears positive but trial fails
- **Mitigation**: Only trade on high-confidence (>0.7) signals, use options to limit downside

### Data Quality Issues
- Scrapers break, missing data
- **Mitigation**: Daily health checks, alerts if data pipelines fail

### Regulatory Risk
- Insider trading concerns
- **Mitigation**: All data sources are public; document data sources and timing

---

## Success Metrics

### System Performance
- **Data freshness**: < 24 hours lag from source update to detection
- **Uptime**: 99%+ for scrapers
- **Signal accuracy**: 70%+ of scored trials match actual outcomes

### Investment Performance
- **Win rate**: 60%+ of trades profitable
- **Average return**: 20%+ per successful trade
- **Sharpe ratio**: > 1.5

---

## Budget

| Item | Cost |
|------|------|
| AWS infrastructure | $150/month |
| OpenAI API | $100/month |
| Database hosting | $50/month |
| Domain + email | $20/month |
| **Total** | **$320/month** |

**One-time setup**: ~$500 (development tools, initial testing)

---

## Timeline

- **Week 1**: Data pipeline for ClinicalTrials.gov + SEC
- **Week 2**: Signal detection + scoring model
- **Week 3**: Email digest + basic dashboard
- **Week 4**: Backtesting on 10 historical trials
- **Week 5+**: Live monitoring + iterative improvements

---

## Next Steps

1. Set up AWS account and PostgreSQL database
2. Build ClinicalTrials.gov scraper (Python + BeautifulSoup/Scrapy)
3. Build SEC EDGAR scraper (Python + SEC API)
4. Implement change detection logic
5. Create scoring model
6. Set up daily email digest (SendGrid)
