# Investment Intelligence Platform - System Architecture

## Overview

The Investment Intelligence Platform is a comprehensive system for detecting investment signals across clinical trials, patents, and insider activity, combined with regime-adaptive trading strategies and risk management.

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              External Data Sources                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ClinicalTrials│  │USPTO/Orange  │  │SEC EDGAR     │  │Market Data           │ │
│  │.gov API      │  │Book          │  │(Form 4, 13F) │  │(Yahoo/Polygon)       │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘ │
└─────────┼─────────────────┼─────────────────┼──────────────────────┼────────────┘
          │                 │                 │                      │
          ▼                 ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Data Ingestion Layer                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Collector Services                           │   │
│  │  • Rate Limiting      • Schema Validation      • Deduplication           │   │
│  │  • Retry Logic        • Error Handling         • Audit Logging           │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐
│  Clinical Trial     │  │  Patent/IP          │  │  Insider Activity           │
│  Signal Engine      │  │  Intelligence       │  │  Signal Engine              │
│                     │  │                     │  │                             │
│ • Trial Ingestion   │  │ • USPTO Ingestion   │  │ • Form 4 Parsing            │
│ • Amendment Track   │  │ • Orange Book Parse │  │ • 13F Holdings              │
│ • Enrollment Signal │  │ • Patent Cliff Calc │  │ • Hiring Signals            │
│ • Outcome Predict   │  │ • Generic Timeline  │  │ • Cluster Detection         │
│ • Signal Scoring    │  │ • Competition Anal  │  │ • Behavioral Scoring        │
└──────────┬──────────┘  └──────────┬──────────┘  └─────────────┬───────────────┘
           │                        │                            │
           └────────────────────────┼────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Signal Aggregation Layer                               │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │                     Combined Signal Scoring Engine                        │   │
│  │  • Multi-Source Weighting    • Confidence Calculation                    │   │
│  │  • Signal Clustering         • Priority Ranking                          │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          ▼                            ▼                            ▼
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐
│  Regime Detection   │  │  Portfolio          │  │  Risk Management            │
│  Service            │  │  Strategy           │  │  Engine                     │
│                     │  │                     │  │                             │
│ • Trend Detection   │  │ • Asset Allocation  │  │ • Position Sizing           │
│ • Volatility Class  │  │ • Rebalancing       │  │ • Circuit Breaker           │
│ • State Transition  │  │ • Signal Response   │  │ • Stop Loss                 │
│ • Multi-Asset       │  │ • Sector Rotation   │  │ • Limit Controls            │
└──────────┬──────────┘  └──────────┬──────────┘  └─────────────┬───────────────┘
           │                        │                            │
           └────────────────────────┼────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           Execution & Monitoring Layer                           │
│  ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────────────┐ │
│  │  Trade Execution   │  │  Dashboard UI      │  │  Alert System              │ │
│  │  (Paper/Live)      │  │  (Streamlit)       │  │  (Email/Slack/SMS)         │ │
│  └────────────────────┘  └────────────────────┘  └────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Architecture

### 1. Clinical Trial Signal Engine

```
clinical_trial_signals/
├── database/
│   ├── schema.sql              # PostgreSQL schema
│   └── db.py                   # Database connection
├── scrapers/
│   ├── clinical_trials_gov.py  # ClinicalTrials.gov API
│   ├── fda_calendar.py         # FDA approval calendar
│   └── pubmed_abstracts.py     # Publication tracking
├── scoring/
│   ├── amendment_scorer.py     # Protocol amendment analysis
│   ├── enrollment_scorer.py    # Enrollment pattern scoring
│   └── composite_scorer.py     # Combined signal scoring
├── alerts/
│   ├── email_alerter.py        # Email notifications
│   └── slack_alerter.py        # Slack notifications
├── main.py                     # Main entry point
└── demo.py                     # Demo mode (no DB)
```

**Data Flow:**
```
ClinicalTrials.gov API
         │
         ▼
┌──────────────────┐
│  Trial Ingestion │──► Raw trial data stored
│  (daily 6 PM)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Amendment Detect │──► Compare with history
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Signal Scoring  │──► Generate composite score
│  (0-10 scale)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Alert Dispatch │──► High-score alerts sent
│   (score >= 7)   │
└──────────────────┘
```

### 2. Patent/IP Intelligence Engine

```
patent_intelligence/
├── sql/
│   └── schema.sql              # Patent database schema
├── extractors/
│   ├── uspto_extractor.py      # USPTO data extraction
│   ├── orange_book_parser.py   # FDA Orange Book
│   └── patent_family.py        # Patent family analysis
├── transformers/
│   ├── expiry_calculator.py    # Patent expiration dates
│   ├── competition_analyzer.py # Generic competition
│   └── timeline_builder.py     # Event timeline
├── loaders/
│   └── postgres_loader.py      # Database loader
└── src/
    └── pipeline.py             # Main ETL pipeline
```

**Data Flow:**
```
USPTO API + Orange Book
         │
         ▼
┌──────────────────┐
│   Data Extract   │──► Patent data + exclusivity
│   (weekly)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Expiry Calc     │──► Calculate cliff dates
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Competition Anal │──► Identify ANDA filings
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Signal Output  │──► Patent cliff opportunities
│                  │
└──────────────────┘
```

### 3. Insider Activity Signal Engine

```
insider_hiring_signals/
├── scrapers/
│   ├── form4_scraper.py        # SEC Form 4 filings
│   ├── thirteenf_scraper.py    # 13F institutional holdings
│   └── linkedin_scraper.py     # Hiring signals
├── models/
│   ├── insider_trade.py        # Insider trade model
│   ├── institutional.py        # 13F holdings model
│   └── hiring_signal.py        # Hiring activity model
├── scoring/
│   ├── insider_scorer.py       # Insider activity score
│   ├── institutional_scorer.py # Institutional interest
│   └── hiring_scorer.py        # Hiring pattern score
├── reports/
│   └── signal_report.py        # Combined reporting
├── main.py                     # Main entry point
└── schema.sql                  # Database schema
```

### 4. Trading System

```
src/umtrading/
├── data/
│   ├── simple_data_collector.py   # Market data collection
│   └── download_historical.py      # Historical data fetch
├── regime/
│   ├── simple_detector.py         # Regime detection
│   └── validate_detector.py       # Validation logic
├── strategies/
│   └── passive_portfolio.py       # Portfolio strategy
├── backtesting/
│   └── simple_backtest.py         # Backtesting engine
├── risk/
│   ├── simple_circuit_breaker.py  # Circuit breaker
│   └── simple_position_sizer.py   # Position sizing
├── dashboards/
│   └── live_monitor.py            # Live monitoring
├── utils/
│   └── daily_logger.py            # Logging
└── integrated_trading_system.py   # Main orchestrator
```

---

## Data Storage Architecture

### PostgreSQL Databases

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PostgreSQL Cluster                           │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│  clinical_trials │ patent_intel   │  insider_signals                 │
│  ───────────────│ ──────────────  │  ──────────────                  │
│  • trials       │ • patents       │  • form4_filings                 │
│  • amendments   │ • exclusivities │  • thirteenf_holdings            │
│  • enrollments  │ • generics      │  • hiring_signals                │
│  • signals      │ • cliff_events  │  • composite_signals             │
│  • alerts       │ • competitors   │  • watchlist                     │
└─────────────────┴─────────────────┴─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         TimescaleDB                                  │
├─────────────────────────────────────────────────────────────────────┤
│  • market_ticks (7-day retention)                                    │
│  • ohlcv_1min (30-day retention)                                     │
│  • ohlcv_5min (90-day retention)                                     │
│  • ohlcv_daily (10-year retention)                                   │
│  • portfolio_history                                                 │
│  • regime_states                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Caching Layer (Redis)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Redis Cache                                │
├─────────────────────────────────────────────────────────────────────┤
│  Hot Data:                                                           │
│  • Current positions                                                 │
│  • Latest prices (5-min TTL)                                         │
│  • Active signals (1-hour TTL)                                       │
│  • Regime state                                                      │
│  • Session state                                                     │
│                                                                      │
│  Rate Limiting:                                                      │
│  • API request counters                                              │
│  • Throttle tokens                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Orchestration Architecture

### Airflow DAGs

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Airflow Scheduler                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  DAG: clinical_trial_pipeline                                        │
│  Schedule: Daily at 6 PM ET                                          │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                 │
│  │ Ingest     │───▶│ Score      │───▶│ Alert      │                 │
│  │ Trials     │    │ Signals    │    │ Dispatch   │                 │
│  └────────────┘    └────────────┘    └────────────┘                 │
│                                                                      │
│  DAG: patent_intelligence_pipeline                                   │
│  Schedule: Weekly on Monday 8 AM ET                                  │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                 │
│  │ Extract    │───▶│ Transform  │───▶│ Load       │                 │
│  │ USPTO      │    │ Cliffs     │    │ DB         │                 │
│  └────────────┘    └────────────┘    └────────────┘                 │
│                                                                      │
│  DAG: insider_signals_pipeline                                       │
│  Schedule: Every 30 min (market hours)                               │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                 │
│  │ Scrape     │───▶│ Score      │───▶│ Update     │                 │
│  │ Form 4     │    │ Activity   │    │ Signals    │                 │
│  └────────────┘    └────────────┘    └────────────┘                 │
│                                                                      │
│  DAG: market_data_pipeline                                           │
│  Schedule: Every 5 min (market hours)                                │
│  ┌────────────┐    ┌────────────┐    ┌────────────┐                 │
│  │ Fetch      │───▶│ Detect     │───▶│ Update     │                 │
│  │ Prices     │    │ Regime     │    │ Portfolio  │                 │
│  └────────────┘    └────────────┘    └────────────┘                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Docker Deployment

```
orchestration/docker/
├── docker-compose.yml          # Full stack definition
├── Dockerfile.pipeline         # Pipeline runner
├── Dockerfile.dashboard        # Streamlit dashboard
├── Dockerfile.airflow          # Airflow scheduler
├── configs/
│   ├── airflow.cfg             # Airflow configuration
│   └── prometheus.yml          # Monitoring config
└── scripts/
    ├── init_databases.sh       # Database initialization
    └── healthcheck.sh          # Health verification
```

```yaml
# docker-compose.yml services
services:
  postgres:          # Primary database
  redis:             # Cache layer
  airflow-webserver: # DAG UI
  airflow-scheduler: # Task scheduler
  pipeline-runner:   # Signal pipelines
  dashboard:         # Streamlit UI
  grafana:           # Monitoring
  prometheus:        # Metrics collection
```

---

## Trading System Architecture

### Integrated Trading System

```
┌─────────────────────────────────────────────────────────────────────┐
│                     IntegratedTradingSystem                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Inputs:                                                             │
│  ├── Market Data (SimpleDataCollector)                               │
│  ├── Regime State (SimpleRegimeDetector)                             │
│  └── Signal Scores (from signal engines)                             │
│                                                                      │
│  Processing:                                                         │
│  ├── Portfolio Strategy (PassivePortfolio)                           │
│  │   └── Regime-adaptive allocation                                  │
│  ├── Position Sizing (SimplePositionSizer)                           │
│  │   └── Kelly criterion + volatility scaling                        │
│  └── Risk Controls (SimpleCircuitBreaker)                            │
│       └── Drawdown limits, stop losses                               │
│                                                                      │
│  Outputs:                                                            │
│  ├── Trade Signals                                                   │
│  ├── Position Updates                                                │
│  ├── Risk Alerts                                                     │
│  └── Performance Logs                                                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Regime Detection Logic

```python
class SimpleRegimeDetector:
    """
    Market Regime States:
    - BULL: SMA50 > SMA200, VIX < 20
    - BEAR: SMA50 < SMA200, VIX > 25
    - HIGH_VOLATILITY: VIX > 30
    - SIDEWAYS: Otherwise
    """

    def detect_regime(self, market_data):
        sma_50 = calculate_sma(market_data, 50)
        sma_200 = calculate_sma(market_data, 200)
        vix = get_current_vix()

        if sma_50 > sma_200 and vix < 20:
            return "BULL"
        elif sma_50 < sma_200 and vix > 25:
            return "BEAR"
        elif vix > 30:
            return "HIGH_VOLATILITY"
        else:
            return "SIDEWAYS"
```

### Risk Management Flow

```
Trade Signal
     │
     ▼
┌────────────────┐     NO      ┌────────────────┐
│ Circuit Breaker├────────────►│ Reject Trade   │
│ Check          │             └────────────────┘
└────────┬───────┘
         │ YES
         ▼
┌────────────────┐
│ Position Sizer │──► Calculate optimal size
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Limit Checks   │──► Verify within limits
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Execute Trade  │──► Place order
└────────┬───────┘
         │
         ▼
┌────────────────┐
│ Log & Monitor  │──► Audit trail
└────────────────┘
```

---

## API Architecture

### REST API Endpoints (FastAPI)

```
/api/v1/
├── signals/
│   ├── GET /clinical-trials          # List clinical trial signals
│   ├── GET /patent-cliffs            # List patent cliff events
│   ├── GET /insider-activity         # List insider signals
│   └── GET /combined                 # Combined scored signals
├── portfolio/
│   ├── GET /positions                # Current positions
│   ├── GET /performance              # Performance metrics
│   ├── POST /rebalance               # Trigger rebalance
│   └── GET /history                  # Portfolio history
├── regime/
│   ├── GET /current                  # Current market regime
│   └── GET /history                  # Regime history
├── alerts/
│   ├── GET /                         # List alerts
│   ├── POST /acknowledge             # Acknowledge alert
│   └── GET /settings                 # Alert settings
└── health/
    └── GET /                         # System health check
```

### WebSocket Feeds

```
ws://server/ws/
├── /market-data     # Real-time price updates
├── /signals         # New signal notifications
├── /regime          # Regime change alerts
├── /portfolio       # Position updates
└── /alerts          # High-priority alerts
```

---

## Security Architecture

### Authentication & Authorization

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Security Layer                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Authentication:                                                     │
│  • API Key authentication for external access                        │
│  • Session tokens for dashboard                                      │
│  • JWT for service-to-service                                        │
│                                                                      │
│  Secrets Management:                                                 │
│  • Environment variables (.env)                                      │
│  • AWS Secrets Manager (production)                                  │
│  • HashiCorp Vault (future)                                          │
│                                                                      │
│  Data Protection:                                                    │
│  • TLS for all external APIs                                         │
│  • Encrypted database connections                                    │
│  • Audit logging for all actions                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Rate Limiting

| API Source | Limit | Window |
|------------|-------|--------|
| Yahoo Finance | 2000 | 1 hour |
| Alpha Vantage | 500 | 1 day |
| SEC EDGAR | 10 | 1 second |
| ClinicalTrials.gov | 100 | 1 minute |

---

## Monitoring Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      Monitoring Stack                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐    │
│  │   Prometheus   │───▶│    Grafana     │───▶│  Alert Manager │    │
│  │   (Metrics)    │    │   (Dashboards) │    │   (Alerts)     │    │
│  └────────────────┘    └────────────────┘    └────────────────┘    │
│                                                                      │
│  Metrics Collected:                                                  │
│  • Pipeline run duration                                             │
│  • Signal generation counts                                          │
│  • API response times                                                │
│  • Database query performance                                        │
│  • Portfolio P&L changes                                             │
│  • Error rates by component                                          │
│                                                                      │
│  Dashboards:                                                         │
│  • System Health                                                     │
│  • Signal Performance                                                │
│  • Trading Activity                                                  │
│  • Pipeline Status                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Architecture

### Development

```
Local Machine
├── Docker Compose (all services)
├── Local PostgreSQL
├── Local Redis
└── Streamlit dashboard
```

### Production (AWS)

```
┌─────────────────────────────────────────────────────────────────────┐
│                           AWS Region                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     VPC (Private Subnet)                       │  │
│  │                                                                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │   EC2      │  │   RDS      │  │   ElastiCache          │  │  │
│  │  │ (Airflow)  │  │ (Postgres) │  │   (Redis)              │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  │                                                                │  │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────────────────┐  │  │
│  │  │  Lambda    │  │    S3      │  │   CloudWatch           │  │  │
│  │  │ (Signals)  │  │ (Storage)  │  │   (Monitoring)         │  │  │
│  │  └────────────┘  └────────────┘  └────────────────────────┘  │  │
│  │                                                                │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                     Public Subnet                              │  │
│  │  ┌────────────┐  ┌────────────────────────────────────────┐  │  │
│  │  │    ALB     │  │         CloudFront (Dashboard)          │  │  │
│  │  │            │  │                                          │  │  │
│  │  └────────────┘  └────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Scalability Considerations

### Horizontal Scaling Points

1. **Signal Pipelines**: Can run multiple instances per signal type
2. **API Servers**: Stateless, can scale behind load balancer
3. **Database**: Read replicas for analytics queries
4. **Cache**: Redis cluster for high availability

### Performance Targets

| Component | Target Latency |
|-----------|---------------|
| Signal generation | < 5 minutes |
| API response | < 200ms |
| Price update | < 1 second |
| Dashboard refresh | < 3 seconds |
| Alert dispatch | < 30 seconds |
