# Insider Activity + Hiring Signals System - Technical Specification

## Overview
Automated system for tracking insider trading, institutional investor movements, and hiring/firing patterns at biotech companies to predict company trajectory and stock movements.

## Business Objective
- **Primary Goal**: Detect material non-public information signals through public data
- **Success Metric**: 65%+ correlation between signal clusters and 30-day stock performance
- **Target ROI**: 15-25% on medium-term (3-6 month) positions

---

## Data Sources

### 1. SEC Form 4 (Insider Trading)
- **Endpoint**: SEC EDGAR API
- **Update Frequency**: Real-time (filed within 2 business days of trade)
- **Key Data**:
  - Transaction type (buy/sell)
  - Number of shares
  - Transaction price
  - Insider title (CEO, CFO, CMO, Director)
  - Ownership stake before/after

### 2. SEC Form 13F (Institutional Holdings)
- **Endpoint**: SEC EDGAR API
- **Update Frequency**: Quarterly (45 days after quarter end)
- **Key Data**:
  - Top biotech investors: Baker Bros, RA Capital, Perceptive, Boxer Capital
  - Position size changes (increased/decreased/new/exited)
  - Put/call positions

### 3. LinkedIn Job Postings
- **Endpoint**: LinkedIn API (requires approval) OR web scraping
- **Update Frequency**: Daily
- **Key Data**:
  - Job titles and departments (R&D, Commercial, Regulatory)
  - Number of open positions
  - Hiring velocity (rate of new postings)
  - Job removals (hiring freeze signals)

### 4. Glassdoor Reviews
- **Endpoint**: Glassdoor API (restricted) OR web scraping
- **Update Frequency**: Weekly
- **Key Data**:
  - Employee sentiment (rating trends)
  - Review text (layoff mentions, pipeline confidence, leadership changes)
  - CEO approval rating

### 5. Clinical Trial Site Jobs
- **Sources**: Indeed, LinkedIn, Glassdoor
- **Update Frequency**: Daily
- **Key Data**:
  - Clinical research coordinator postings at trial sites
  - Hiring freezes at sites (enrollment problems)

### 6. SEC Form 8-K (Material Events)
- **Endpoint**: SEC EDGAR API
- **Update Frequency**: Real-time
- **Key Data**:
  - Executive departures (Item 5.02)
  - Material agreements (partnerships, licensing deals)
  - Results of operations (early trial data)

---

## Signal Categories & Weights

### Category 1: Insider Trading Signals

#### Bullish Signals
| Signal | Weight | Interpretation |
|--------|--------|----------------|
| CEO buys >$100K | +5 | Extreme confidence (rare in biotech) |
| CFO buys >$50K | +4 | Financial health + positive outlook |
| CMO (Chief Medical Officer) buys | +5 | Clinical data confidence |
| Multiple insiders buy in 30 days | +6 | Coordinated optimism |
| Director buys after joining board | +3 | Due diligence revealed opportunity |

#### Bearish Signals
| Signal | Weight | Interpretation |
|--------|--------|----------------|
| CEO sells >10% of holdings | -4 | Lack of confidence or need cash |
| CFO sells before quarter end | -5 | Knows financials will disappoint |
| Multiple insiders sell in 30 days | -6 | Coordinated pessimism |
| C-suite selling after trial update | -5 | Data not as good as market thinks |

**Important**: Distinguish between:
- **Rule 10b5-1 plan** (pre-scheduled, less informative)
- **Open market purchase** (discretionary, highly informative)

### Category 2: Institutional Investor Signals

#### Bullish Signals
| Signal | Weight | Interpretation |
|--------|--------|----------------|
| Baker Bros new position | +5 | Top biotech fund sees value |
| RA Capital increases >50% | +4 | High conviction increase |
| 3+ top funds initiate positions | +6 | Smart money convergence |
| Activist investor (Icahn, etc.) files 13D | +7 | Potential catalyst coming |

#### Bearish Signals
| Signal | Weight | Interpretation |
|--------|--------|----------------|
| Top fund exits completely | -5 | Lost confidence |
| 3+ funds reduce positions | -6 | Smart money fleeing |
| Perceptive (large fund) cuts by >50% | -4 | Major concerns |

**Note**: 13F data is 45+ days delayed, so combine with Form 4 for timeliness

### Category 3: Hiring Pattern Signals

#### Bullish Signals
| Signal | Weight | Interpretation |
|--------|--------|----------------|
| 5+ commercial roles posted | +5 | Expecting drug approval |
| VP of Manufacturing hired | +4 | Scaling for commercialization |
| Regulatory affairs team expansion | +3 | Preparing FDA submission |
| Clinical trial coordinators hired at sites | +4 | Enrollment accelerating |
| Medical affairs roles (MSLs) | +5 | Pre-launch prep |

#### Bearish Signals
| Signal | Weight | Interpretation |
|--------|--------|----------------|
| Job postings removed en masse | -4 | Hiring freeze (cash concerns) |
| CMO departure (Form 8-K) | -6 | Pipeline or data issues |
| CFO departure (Form 8-K) | -5 | Financial problems or board conflict |
| Glassdoor mentions "layoffs" spike | -4 | Restructuring coming |
| Clinical site hiring stops | -5 | Enrollment problems |

### Category 4: Sentiment Signals

#### Glassdoor Sentiment Analysis
```python
# AI-powered text analysis of reviews
sentiment_score = analyze_reviews(company, last_90_days)

if sentiment_score < 2.5:  # Out of 5
    signal_weight = -3  # "Management doesn't believe in pipeline"
elif sentiment_score > 4.0:
    signal_weight = +2  # "Excited about upcoming data"
else:
    signal_weight = 0
```

---

## Data Pipeline Architecture

### Stack
- **Language**: Python 3.11+
- **Data Storage**: PostgreSQL + Redis (caching)
- **ETL**: Prefect for orchestration
- **Cloud**: AWS (Lambda for scrapers, RDS, SNS for alerts)
- **AI**: Claude Sonnet for sentiment analysis of reviews/filings
- **Visualization**: Streamlit dashboard

### Pipeline Flow
```
┌─────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                      │
├─────────────────────────────────────────────────────────────┤
│  Form 4    13F    LinkedIn    Glassdoor    8-K    Indeed    │
│    ↓        ↓        ↓           ↓          ↓        ↓      │
│  [RT]    [Qtrly]  [Daily]     [Weekly]    [RT]   [Daily]   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  Data Processing Layer                       │
├─────────────────────────────────────────────────────────────┤
│  • Insider transaction parsing (buy vs 10b5-1 plan)          │
│  • 13F change detection (position size deltas)               │
│  • Job posting classification (R&D vs commercial vs admin)   │
│  • Sentiment analysis (Glassdoor reviews)                    │
│  • Executive departure classification (voluntary vs fired)   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      Signal Aggregation                      │
├─────────────────────────────────────────────────────────────┤
│  • Composite score per company (-20 to +20)                  │
│  • Time-decay weighting (recent signals > old)               │
│  • Clustering detection (multiple signals same direction)    │
│  • Outlier detection (unusual patterns)                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     Output & Alerts                          │
├─────────────────────────────────────────────────────────────┤
│  • Real-time alerts (Form 4 filed within 30 min)             │
│  • Daily digest (top 5 bullish/bearish signal clusters)      │
│  • Weekly 13F analysis (smart money moves)                   │
│  • Dashboard with company watchlist                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Scoring Model

### Composite Signal Score

```python
def calculate_signal_score(company_ticker, lookback_days=90):
    signals = get_signals(company_ticker, lookback_days)

    score = 0
    confidence = 0

    for signal in signals:
        # Time decay: recent signals weighted more heavily
        days_ago = (today - signal.date).days
        decay_factor = 1 / (1 + days_ago/30)  # 30-day half-life

        weighted_score = signal.weight * decay_factor
        score += weighted_score
        confidence += abs(signal.weight) * decay_factor

    # Normalize to -10 to +10 scale
    normalized_score = max(-10, min(10, score / 2))

    # Confidence based on signal count and recency
    confidence_score = min(1.0, confidence / 20)

    return {
        'score': normalized_score,
        'confidence': confidence_score,
        'signal_count': len(signals),
        'recommendation': get_recommendation(normalized_score, confidence_score)
    }

def get_recommendation(score, confidence):
    if score >= 6 and confidence >= 0.7:
        return "STRONG BUY"
    elif score >= 3 and confidence >= 0.5:
        return "BUY"
    elif score <= -6 and confidence >= 0.7:
        return "STRONG SELL / SHORT"
    elif score <= -3 and confidence >= 0.5:
        return "SELL / AVOID"
    else:
        return "NEUTRAL / MONITOR"
```

### Example Calculation

**Company: XYZ Biotech**

| Signal | Date | Weight | Days Ago | Decay Factor | Contribution |
|--------|------|--------|----------|--------------|--------------|
| CEO bought $200K | 10 days ago | +5 | 10 | 0.75 | +3.75 |
| Baker Bros increased 60% | 50 days ago | +4 | 50 | 0.38 | +1.52 |
| 8 commercial jobs posted | 15 days ago | +5 | 15 | 0.67 | +3.35 |
| CMO hired | 5 days ago | +4 | 5 | 0.86 | +3.44 |

**Total Score**: +12.06 / 2 = **+6.03** (Strong Buy)
**Confidence**: 12.06 / 20 = **0.60** (Medium-High)

---

## Database Schema

### insider_transactions
```sql
CREATE TABLE insider_transactions (
    transaction_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10),
    insider_name VARCHAR(200),
    insider_title VARCHAR(100),
    transaction_date DATE,
    transaction_type VARCHAR(20),  -- Purchase, Sale
    shares INT,
    price_per_share DECIMAL(10,2),
    transaction_value BIGINT,
    is_10b5_1_plan BOOLEAN,
    ownership_after_pct DECIMAL(5,2),
    filing_date DATE,
    signal_weight INT
);
```

### institutional_holdings
```sql
CREATE TABLE institutional_holdings (
    holding_id SERIAL PRIMARY KEY,
    fund_name VARCHAR(200),
    company_ticker VARCHAR(10),
    quarter DATE,
    shares BIGINT,
    value BIGINT,
    pct_change_vs_prev_quarter DECIMAL(5,2),
    is_new_position BOOLEAN,
    is_exit BOOLEAN,
    signal_weight INT
);
```

### job_postings
```sql
CREATE TABLE job_postings (
    job_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10),
    job_title VARCHAR(300),
    department VARCHAR(100),  -- R&D, Commercial, Regulatory, etc.
    post_date DATE,
    removal_date DATE,
    source VARCHAR(50),  -- LinkedIn, Indeed, etc.
    is_senior_role BOOLEAN,
    signal_weight INT
);
```

### executive_changes
```sql
CREATE TABLE executive_changes (
    change_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10),
    executive_name VARCHAR(200),
    title VARCHAR(100),
    change_type VARCHAR(20),  -- Departure, Hire
    effective_date DATE,
    announcement_date DATE,
    reason TEXT,  -- Parsed from 8-K
    signal_weight INT
);
```

### glassdoor_sentiment
```sql
CREATE TABLE glassdoor_sentiment (
    review_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10),
    review_date DATE,
    rating DECIMAL(2,1),
    review_text TEXT,
    sentiment_score DECIMAL(3,2),  -- -1 to +1
    mentions_layoffs BOOLEAN,
    mentions_pipeline BOOLEAN,
    signal_weight INT
);
```

### signal_scores
```sql
CREATE TABLE signal_scores (
    score_id SERIAL PRIMARY KEY,
    company_ticker VARCHAR(10),
    score_date DATE,
    composite_score DECIMAL(4,2),  -- -10 to +10
    confidence DECIMAL(3,2),  -- 0 to 1
    signal_count INT,
    recommendation VARCHAR(50),
    contributing_signals JSONB
);
```

---

## AI-Powered Features

### 1. Glassdoor Review Sentiment Analysis

```python
prompt = f"""
Analyze these Glassdoor reviews for {company_name}:

{reviews_text}

Extract:
1. Overall sentiment (-1 to +1)
2. Pipeline confidence mentions (quote examples)
3. Layoff/restructuring mentions
4. Management quality sentiment
5. Any material information about upcoming events

Format: JSON
"""

response = claude_sonnet.complete(prompt)
```

### 2. Form 8-K Executive Departure Analysis

```python
prompt = f"""
This biotech company filed an 8-K announcing CMO departure:

{filing_text}

Determine:
1. Was this voluntary or involuntary?
2. Does the language suggest conflict (e.g., "pursue other opportunities" vs "personal reasons")?
3. Is there evidence of planned succession vs emergency replacement?
4. Severity rating (1-10): How concerning is this for investors?

Provide 1-sentence reasoning.
"""
```

---

## MVP Scope (Week 1-2)

### Must Have
- [x] Form 4 scraper for top 100 biotech companies
- [x] 13F tracking for top 10 biotech-focused funds
- [x] LinkedIn job scraper (or API if approved)
- [x] Basic signal scoring model
- [x] Daily email digest

### Nice to Have (Future)
- [ ] Glassdoor sentiment analysis
- [ ] Form 8-K executive departure parsing
- [ ] Clinical trial site job tracking
- [ ] Web dashboard
- [ ] Real-time Slack alerts

---

## Watchlist Configuration

### Initial Watchlist (Top 50 Biotechs)
**Criteria**:
- Market cap: $500M - $5B (volatility sweet spot)
- Has active clinical trials (material events expected)
- Public for >1 year (history available)
- Average volume >500K shares/day (liquid)

---

## Risk Management

### False Positives
- Insider buys for reasons unrelated to company (personal rebalancing)
- **Mitigation**: Only flag discretionary purchases >$50K

### Lagging Data
- 13F is 45+ days old
- **Mitigation**: Combine with real-time Form 4 for triangulation

### Hiring Noise
- Job postings don't always mean imminent commercialization
- **Mitigation**: Require cluster of 3+ signals in same direction

---

## Success Metrics

### System Performance
- **Form 4 detection latency**: < 30 minutes
- **Job posting coverage**: 90%+ of LinkedIn/Indeed postings
- **Signal accuracy**: 65%+ correlation with 30-day stock moves

### Investment Performance
- **Win rate**: 60%+ of flagged opportunities profitable
- **Average return**: 18%+ per position
- **Sharpe ratio**: > 1.2

---

## Budget

| Item | Cost |
|------|------|
| LinkedIn Recruiter Lite (job access) | $140/month |
| AWS infrastructure | $100/month |
| AI API (sentiment analysis) | $150/month |
| Web scraping proxies | $50/month |
| **Total** | **$440/month** |

**One-time setup**: ~$600

---

## Timeline

- **Week 1**: Form 4 + 13F scrapers
- **Week 2**: LinkedIn job scraper + scoring model
- **Week 3**: Email digest + basic dashboard
- **Week 4**: Backtesting on 2023 signals
- **Week 5+**: Live monitoring + Glassdoor integration

---

## Next Steps

1. Set up SEC EDGAR RSS feed for Form 4 alerts
2. Build LinkedIn job scraper (or apply for API access)
3. Create database schema
4. Implement signal scoring model
5. Generate first daily digest
