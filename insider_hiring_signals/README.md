# Insider Activity + Hiring Signals System

Automated system for tracking insider trading, institutional investor movements, and hiring/firing patterns at biotech companies to predict company trajectory and stock movements.

## Overview

This system aggregates multiple data sources to generate composite signal scores for biotech companies:

- **SEC Form 4**: Insider trading transactions (CEO/CFO/CMO purchases and sales)
- **SEC 13F**: Institutional investor holdings (quarterly position changes)
- **Job Postings**: Hiring patterns from company career pages
- **Signal Scoring**: Time-decay weighted composite scores

## Features

- SEC Form 4 scraper with 10b5-1 plan detection
- SEC 13F tracking for top 10 biotech-focused institutional funds
- Job posting scraper supporting Greenhouse, Lever, and generic career pages
- Time-decay weighted signal scoring model
- Daily email digest with top bullish/bearish signals
- Console, HTML, and JSON report generation
- PostgreSQL database for data persistence

## Project Structure

```
insider_hiring_signals/
|-- config/
|   |-- config.yaml.template    # Configuration template
|-- data/
|   |-- reports/                # Generated reports
|-- models/
|   |-- __init__.py
|   |-- signal_scorer.py        # Signal scoring model
|-- reports/
|   |-- __init__.py
|   |-- email_digest.py         # Daily email digest
|   |-- report_generator.py     # Report generation
|-- scrapers/
|   |-- __init__.py
|   |-- form4_scraper.py        # SEC Form 4 scraper
|   |-- form13f_scraper.py      # SEC 13F scraper
|   |-- job_scraper.py          # Job posting scraper
|-- utils/
|   |-- __init__.py
|   |-- config.py               # Configuration loader
|   |-- database.py             # Database utilities
|   |-- logger.py               # Logging setup
|-- main.py                     # Main entry point
|-- schema.sql                  # Database schema
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
```

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- SEC EDGAR account (for API access)

## Installation

### 1. Clone and Setup Environment

```bash
cd insider_hiring_signals

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure PostgreSQL

```bash
# Create database
psql -U postgres
CREATE DATABASE insider_signals;
CREATE USER insider_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE insider_signals TO insider_user;
\q
```

### 3. Initialize Database Schema

```bash
# Using psql
psql -U insider_user -d insider_signals -f schema.sql

# Or using the CLI
python main.py init-db
```

### 4. Configure Application

```bash
# Copy template
cp config/config.yaml.template config/config.yaml

# Edit configuration
nano config/config.yaml
```

Key configuration items:

```yaml
# Database connection
database:
  host: localhost
  port: 5432
  name: insider_signals
  user: insider_user
  password: your_password

# SEC EDGAR requires a User-Agent with your email
sec_edgar:
  user_agent: "YourName your.email@example.com"
  rate_limit_requests_per_second: 10

# Email for daily digest
email:
  smtp_server: smtp.gmail.com
  smtp_port: 587
  sender_email: your_email@gmail.com
  sender_password: your_app_password  # Use App Password for Gmail
  recipients:
    - recipient@example.com
```

### Environment Variables (Alternative)

You can also configure via environment variables:

```bash
export DATABASE_HOST=localhost
export DATABASE_PORT=5432
export DATABASE_NAME=insider_signals
export DATABASE_USER=insider_user
export DATABASE_PASSWORD=your_password
export SEC_USER_AGENT="YourName your.email@example.com"
export ANTHROPIC_API_KEY=your_key  # For sentiment analysis (optional)
```

## Usage

### Command Line Interface

```bash
# Run all scrapers
python main.py scrape --all

# Run specific scrapers
python main.py scrape --form4
python main.py scrape --13f
python main.py scrape --jobs

# Run signal scoring
python main.py score                    # Score all watchlist companies
python main.py score --ticker MRNA      # Score specific company

# Generate reports
python main.py report --console         # Print to terminal
python main.py report --html            # Generate HTML report
python main.py report --json            # Export as JSON
python main.py report --example         # Generate example report

# Daily digest
python main.py digest --save            # Save digest to file
python main.py digest --send            # Send email digest

# Run complete daily pipeline
python main.py run-daily
```

### Programmatic Usage

```python
from scrapers import Form4Scraper, Form13FScraper, JobScraper
from models import SignalScorer
from reports import EmailDigest, ReportGenerator

# Scrape Form 4 filings
scraper = Form4Scraper()
result = scraper.run(tickers=['MRNA', 'VRTX'], days=30)

# Calculate signal scores
scorer = SignalScorer()
score = scorer.calculate_score('MRNA')
print(f"Score: {score.composite_score}, Rec: {score.recommendation}")

# Generate report
generator = ReportGenerator()
generator.print_console_report()
```

## Signal Scoring Model

### Composite Score Calculation

The system calculates a composite score from -10 to +10 using:

1. **Raw Signal Weights**: Based on signal type and magnitude
2. **Time Decay**: Recent signals weighted more heavily (30-day half-life)
3. **Normalization**: Divided by 2 and clamped to [-10, +10]

```
decay_factor = 1 / (1 + days_ago / 30)
weighted_score = raw_weight * decay_factor
composite_score = clamp(sum(weighted_scores) / 2, -10, +10)
```

### Signal Weights

**Bullish Signals (Positive)**:
| Signal | Weight | Description |
|--------|--------|-------------|
| CEO_PURCHASE | +5 | CEO buys >$100K open market |
| CMO_PURCHASE | +5 | Chief Medical Officer buys |
| MULTIPLE_INSIDER_BUY | +6 | 2+ insiders buy in 30 days |
| FUND_NEW_POSITION | +5 | Top fund initiates position |
| FUND_INCREASE_50 | +4 | Fund increases >50% |
| COMMERCIAL_BUILDUP | +5 | 5+ commercial roles posted |
| MSL_HIRING | +5 | Medical Science Liaison roles |

**Bearish Signals (Negative)**:
| Signal | Weight | Description |
|--------|--------|-------------|
| CEO_SALE | -4 | CEO sells >10% holdings |
| CFO_SALE | -5 | CFO sells (financial concern) |
| MULTIPLE_INSIDER_SELL | -6 | 2+ insiders sell in 30 days |
| FUND_EXIT | -5 | Top fund exits completely |
| FUND_DECREASE_50 | -4 | Fund decreases >50% |
| HIRING_FREEZE | -4 | Job postings removed |

### Recommendations

| Recommendation | Score | Confidence |
|----------------|-------|------------|
| STRONG BUY | >= 6.0 | >= 0.70 |
| BUY | >= 3.0 | >= 0.50 |
| NEUTRAL | Between | Any |
| SELL | <= -3.0 | >= 0.50 |
| STRONG SELL | <= -6.0 | >= 0.70 |

## Data Sources

### SEC EDGAR

- **Form 4**: Filed within 2 business days of insider transaction
- **Form 13F**: Filed within 45 days of quarter end
- **Rate Limit**: 10 requests per second (SEC requirement)
- **User-Agent**: Required with your email address

### Job Postings

Supported platforms:
- Greenhouse (API-based)
- Lever (API-based)
- Generic career pages (HTML parsing)

### Tracked Institutional Investors

The system tracks the top biotech-focused funds:
1. Baker Bros Advisors LP
2. RA Capital Management LP
3. Perceptive Advisors LLC
4. Boxer Capital LLC
5. OrbiMed Advisors LLC
6. Farallon Capital Management LLC
7. Viking Global Investors LP
8. Partner Fund Management LP
9. Deerfield Management Company
10. Tang Capital Management LLC

## Watchlist Configuration

Edit `config/config.yaml` to customize the watchlist:

```yaml
watchlist:
  - MRNA
  - VRTX
  - REGN
  - CRSP
  - BEAM
  # ... add more tickers
```

## Scheduling

### Using Cron (Linux/Mac)

```bash
# Run daily at 7 AM
0 7 * * * /path/to/venv/bin/python /path/to/main.py run-daily

# Run scrapers every 6 hours
0 */6 * * * /path/to/venv/bin/python /path/to/main.py scrape --all
```

### Using Windows Task Scheduler

Create a batch file `run_daily.bat`:
```batch
@echo off
cd C:\path\to\insider_hiring_signals
call venv\Scripts\activate
python main.py run-daily
```

## Database Schema

### Core Tables

- `companies`: Company reference data
- `insider_transactions`: Form 4 insider trades
- `institutional_holdings`: 13F holdings data
- `job_postings`: Job posting data
- `signal_scores`: Calculated signal scores
- `signals`: Individual contributing signals
- `email_digests`: Digest history

### Views

- `v_recent_insider_activity`: Last 30 days of insider trades
- `v_institutional_changes`: Latest quarter position changes
- `v_current_signals`: Most recent signal scores
- `v_job_trends`: Job posting trends by company

## Troubleshooting

### Common Issues

**SEC Rate Limiting**
```
Error: 429 Too Many Requests
```
Solution: Increase `rate_limit_seconds` in config or wait before retrying.

**Database Connection**
```
Error: connection refused
```
Solution: Ensure PostgreSQL is running and credentials are correct.

**No Data Returned**
- Check that tickers are valid
- Verify SEC EDGAR User-Agent is set
- Ensure database schema is initialized

### Logging

Logs are written to `logs/insider_signals.log`. Adjust log level in config:

```yaml
logging:
  level: DEBUG  # DEBUG, INFO, WARNING, ERROR
```

## Disclaimer

This system is for informational purposes only and does not constitute investment advice. Key limitations:

- **13F Data Delay**: 45+ days old when filed
- **Job Posting Noise**: May not reflect actual hiring decisions
- **Insider Trade Reasons**: May be personal, not company-related
- **Historical Correlation**: Past patterns don't guarantee future results

Always conduct your own due diligence before making investment decisions.

## License

MIT License - See LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## Support

For issues or questions, please open a GitHub issue.
