# Clinical Trial Signal Detection System

An automated intelligence system for detecting early signals of clinical trial outcomes before public announcements. Provides advance warning on trial results for investment analysis.

## Features

- **ClinicalTrials.gov Scraper**: Fetches Phase 3 trial data including enrollment, status, endpoints
- **SEC EDGAR Scraper**: Monitors 8-K filings from biotech companies for material events
- **Change Detection**: Identifies protocol amendments, enrollment changes, timeline shifts
- **Signal Scoring**: Multi-factor model to score trial outcomes on 0-10 scale
- **Email Digest**: Daily summary of detected signals and recommendations

## Project Structure

```
clinical_trial_signals/
├── main.py                    # Main orchestration script
├── config.py                  # Configuration settings
├── config_template.env        # Environment variable template
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── database/
│   ├── __init__.py
│   ├── connection.py          # Database connection management
│   ├── models.py              # Data models (Trial, Signal, Score, etc.)
│   └── schema.sql             # PostgreSQL schema
├── scrapers/
│   ├── __init__.py
│   ├── clinicaltrials.py      # ClinicalTrials.gov API scraper
│   └── sec_edgar.py           # SEC EDGAR API scraper
├── scoring/
│   ├── __init__.py
│   └── signal_scorer.py       # Signal scoring model
├── alerts/
│   ├── __init__.py
│   └── email_digest.py        # Email digest generation and sending
└── utils/
    ├── __init__.py
    └── change_detection.py    # Trial change detection
```

## Prerequisites

- Python 3.11+
- PostgreSQL 14+
- SendGrid account (for email alerts, optional)

## Installation

### 1. Clone and Install Dependencies

```bash
cd clinical_trial_signals
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the template and fill in your values:

```bash
cp config_template.env .env
```

Edit `.env` with your settings:

```env
# Database Configuration (PostgreSQL)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=clinical_trials
DB_USER=postgres
DB_PASSWORD=your_password_here

# Email Configuration (SendGrid) - Optional
EMAIL_ENABLED=false
SENDGRID_API_KEY=your_sendgrid_api_key_here
FROM_EMAIL=alerts@yourdomain.com
TO_EMAILS=analyst1@example.com,analyst2@example.com

# SEC EDGAR Configuration
# IMPORTANT: SEC requires a valid User-Agent with contact info
SEC_USER_AGENT=YourCompany research@yourdomain.com

# Logging
LOG_LEVEL=INFO
LOG_FILE=clinical_trials.log

# Run Mode
DRY_RUN=false
```

### 3. Setup PostgreSQL Database

Create the database:

```bash
createdb clinical_trials
```

Initialize the schema:

```bash
python main.py --init-db
```

Or manually:

```bash
psql -d clinical_trials -f database/schema.sql
```

## Usage

### Full Pipeline (Recommended for Daily Use)

Run the complete pipeline - fetches data, detects changes, calculates scores, sends email:

```bash
python main.py --full
```

### Individual Stages

```bash
# Only fetch trial data from ClinicalTrials.gov and SEC
python main.py --fetch --trials 20

# Only run change detection on existing data
python main.py --detect

# Only calculate scores
python main.py --score

# Only send email digest
python main.py --email
```

### Dry Run (Testing)

Test without database writes or sending emails:

```bash
python main.py --dry-run --full
```

### Command Line Options

```
Options:
  --full        Run full pipeline (default)
  --fetch       Only fetch new data from sources
  --detect      Only run change detection
  --score       Only calculate scores
  --email       Only send email digest
  --init-db     Initialize database schema
  --dry-run     Run without database writes or sending emails
  --log-level   DEBUG, INFO, WARNING, ERROR (default: INFO)
  --trials N    Number of trials to fetch (default: 20)
  --days N      Days to look back for filings/signals (default: 7)
```

## Signal Types and Weights

### Positive Signals (Trial Likely to Succeed)

| Signal | Weight | Description |
|--------|--------|-------------|
| Sites Added | +3 | New trial sites indicate enrollment going well |
| Early Enrollment | +3 | Enrollment completed ahead of schedule |
| Completion Accelerated | +3 | Trial completion date moved earlier |
| SEC 8-K Positive | +3 | Positive trial mentions in SEC filings |
| Status Change Positive | +2 | Trial status changed favorably |
| Enrollment Increase | +2 | Significant increase in enrolled patients |

### Negative Signals (Trial Likely to Fail)

| Signal | Weight | Description |
|--------|--------|-------------|
| Endpoint Change | -5 | Primary endpoint modified (major red flag) |
| Sites Removed | -4 | Trial sites removed indicates problems |
| Completion Delayed | -3 | Trial completion date pushed back |
| SEC 8-K Negative | -3 | Negative trial mentions in SEC filings |
| Enrollment Extended | -3 | Enrollment timeline extended |
| Status Change Negative | -2 | Trial suspended/terminated |
| Enrollment Decrease | -2 | Unusual decrease in enrollment |

## Scoring Model

The composite score ranges from 0-10:

- **8-10**: Strong positive signals (STRONG_BUY)
- **5-7**: Moderate positive signals (BUY)
- **4-5**: Neutral/mixed signals (HOLD)
- **3-4**: Moderate negative signals (SHORT)
- **0-3**: Strong negative signals (STRONG_SHORT)

Confidence is calculated based on:
- Number of signals (more = higher confidence)
- Signal diversity (multiple types = higher confidence)
- Signal consistency (all same direction = higher confidence)
- Historical accuracy

## Scheduling Daily Runs

### Using Cron (Linux/macOS)

Add to crontab (`crontab -e`):

```bash
# Run daily at 6 AM
0 6 * * * cd /path/to/clinical_trial_signals && /path/to/venv/bin/python main.py --full >> /var/log/clinical_trials.log 2>&1
```

### Using Task Scheduler (Windows)

Create a scheduled task to run:

```batch
C:\path\to\venv\Scripts\python.exe C:\path\to\clinical_trial_signals\main.py --full
```

### Using systemd Timer (Linux)

Create `/etc/systemd/system/clinical-trials.service`:

```ini
[Unit]
Description=Clinical Trial Signal Detection

[Service]
Type=oneshot
WorkingDirectory=/path/to/clinical_trial_signals
ExecStart=/path/to/venv/bin/python main.py --full
User=your_user
```

Create `/etc/systemd/system/clinical-trials.timer`:

```ini
[Unit]
Description=Run Clinical Trial Signal Detection Daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true

[Install]
WantedBy=timers.target
```

Enable: `sudo systemctl enable --now clinical-trials.timer`

## Database Tables

### trials
Stores clinical trial information from ClinicalTrials.gov

### trial_signals
Detected signals with type, weight, and source

### trial_scores
Daily composite scores and recommendations

### companies
Biotech companies being monitored (with tickers and CIKs)

### sec_filings
SEC EDGAR filings processed

### trial_history
Historical snapshots for change detection

## API Data Sources

### ClinicalTrials.gov API v2
- Base URL: https://clinicaltrials.gov/api/v2
- No authentication required
- Rate limit: ~3 requests/second recommended

### SEC EDGAR API
- Base URL: https://data.sec.gov
- Requires User-Agent header with contact info
- Rate limit: 10 requests/second max

## Cost Estimates

| Item | Monthly Cost |
|------|--------------|
| PostgreSQL (local or free tier) | $0-50 |
| SendGrid (free tier: 100 emails/day) | $0 |
| Compute (local or small VM) | $0-50 |
| **Total** | **$0-100/month** |

## Extending the System

### Adding New Signal Types

1. Define signal weight in `config.py`:
```python
"new_signal_type": 3,  # or negative for bearish signals
```

2. Add detection logic in appropriate scraper or `change_detection.py`

3. Create `TrialSignal` and save to database

### Adding New Data Sources

1. Create new scraper in `scrapers/` directory
2. Implement `fetch_*` method following existing patterns
3. Add to `main.py` pipeline

### Custom Scoring

Modify `scoring/signal_scorer.py`:
- Adjust weight calculations in `score_trial()`
- Modify confidence factors in `_calculate_confidence()`
- Adjust thresholds in `_get_recommendation()`

## Troubleshooting

### Database Connection Errors

```
psycopg2.OperationalError: could not connect to server
```

- Verify PostgreSQL is running: `pg_isready`
- Check connection settings in `.env`
- Ensure database exists: `psql -l | grep clinical_trials`

### SEC API Errors

```
403 Forbidden
```

- Ensure `SEC_USER_AGENT` is set with valid contact email
- SEC requires: "Company Name contact@email.com"

### No Signals Detected

- Check if trials exist: `SELECT COUNT(*) FROM trials;`
- Verify trial status is active (not COMPLETED/TERMINATED)
- Run with `--log-level DEBUG` for more detail

### Email Not Sending

- Verify `EMAIL_ENABLED=true` in `.env`
- Check SendGrid API key is valid
- Verify sender email is verified in SendGrid

## Disclaimer

This system is for informational and research purposes only. It is NOT financial advice. All investment decisions should be made after conducting your own research. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred from using this system.

## License

MIT License - See LICENSE file for details.
