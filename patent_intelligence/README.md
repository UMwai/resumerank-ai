# Patent/IP Intelligence System

Automated system for tracking pharmaceutical patent expirations, generic drug approvals, and patent cliff events to identify investment opportunities.

## Overview

This system provides:
- **FDA Orange Book data ingestion**: Extracts drug and patent information from the FDA Orange Book
- **USPTO patent tracking**: Calculates patent expiration dates including PTA/PTE adjustments
- **ANDA monitoring**: Tracks generic drug applications and approvals
- **Patent cliff calendar**: 12-month forward view of upcoming patent expirations
- **Certainty scoring**: Evaluates likelihood of patent cliff events occurring
- **Weekly email digest**: Automated notifications of upcoming events

## Features

### Data Sources
- FDA Orange Book (drugs, patents, exclusivity periods)
- USPTO PatentsView API (patent details, assignees)
- FDA ANDA database (generic drug applications)

### Scoring Model
The certainty score (0-100%) is calculated based on:
- Patent expiration status (40% weight)
- Active litigation (30% weight)
- ANDA approvals (20% weight)
- Extension likelihood (10% weight)

### Market Opportunity Classification
- **Blockbuster**: >$1B revenue at risk
- **High Value**: $500M-$1B revenue at risk
- **Medium Value**: $100M-$500M revenue at risk
- **Small**: <$100M revenue at risk

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 14+
- pip package manager

### Setup

1. Clone the repository and navigate to the project:
```bash
cd patent_intelligence
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up configuration:
```bash
# Copy template files
cp config/.env.template config/.env
cp config/config.yaml config/config.local.yaml

# Edit .env with your credentials
# Edit config.local.yaml for any custom settings
```

5. Set up the database:
```bash
# Create database
createdb patent_intelligence

# Create user (optional)
psql -c "CREATE USER patent_user WITH PASSWORD 'your_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE patent_intelligence TO patent_user;"

# Run schema migrations
psql -d patent_intelligence -f sql/schema.sql
```

## Configuration

### Environment Variables (.env)
```
# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=patent_intelligence
DB_USER=patent_user
DB_PASSWORD=your_secure_password

# Email (for weekly digest)
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_app_specific_password
EMAIL_RECIPIENT_1=recipient@example.com
```

### Config File (config.yaml)
Key settings:
- `etl.top_drugs_count`: Number of drugs to track (default: 50)
- `calendar.forward_months`: Calendar lookahead period (default: 12)
- `scoring.*`: Scoring model weights
- `email.enabled`: Enable/disable email notifications

## Usage

### Running the Pipeline

**Full pipeline (recommended)**:
```bash
python -m src.pipeline --top-drugs 50 --output-dir output
```

**Without database loading** (useful for testing):
```bash
python -m src.pipeline --no-db --output-dir output
```

**With email notifications**:
```bash
python -m src.pipeline --send-email
```

### Command Line Options
```
--config PATH       Path to config file
--top-drugs N       Number of drugs to process (default: 50)
--no-db             Skip database loading
--send-email        Send weekly digest email
--no-export         Skip file export
--output-dir DIR    Output directory (default: output)
```

### Programmatic Usage

```python
from src.pipeline import PatentIntelligencePipeline

# Initialize pipeline
pipeline = PatentIntelligencePipeline()

# Run full pipeline
results = pipeline.run_full_pipeline(
    top_n=50,
    load_db=True,
    send_email=False,
    export_files=True,
    output_dir="output"
)

# Access results
print(f"Drugs processed: {results['drugs_extracted']}")
print(f"Events generated: {results['events_generated']}")

# Clean up
pipeline.close()
```

### Using Individual Components

**Extract Orange Book Data**:
```python
from src.extractors.orange_book import OrangeBookExtractor

extractor = OrangeBookExtractor()
drugs, patents, exclusivity = extractor.extract_for_database(top_n=50)
```

**Calculate Patent Expirations**:
```python
from src.extractors.uspto import PatentExpirationCalculator
from datetime import date

calculator = PatentExpirationCalculator()
result = calculator.calculate_expiration(
    filing_date=date(2010, 6, 15),
    pta_days=180,
    pte_days=365,
)
print(f"Final expiration: {result['final_expiration_date']}")
```

**Score a Patent Cliff**:
```python
from src.transformers.scoring import PatentCliffScorer, DrugPatentData
from datetime import date

scorer = PatentCliffScorer()
drug_data = DrugPatentData(
    drug_id=1,
    brand_name="DrugX",
    generic_name="drugx",
    branded_company="PharmaCo",
    branded_company_ticker="PHRM",
    annual_revenue=5_000_000_000,
    patent_numbers=["US1234567"],
    earliest_expiration=date(2025, 6, 15),
    latest_expiration=date(2025, 6, 15),
    all_patents_expired=False,
    expiring_patents_count=1,
    total_patents_count=1,
    active_litigation_count=0,
    resolved_litigation_count=0,
    patents_invalidated=0,
    approved_generics_count=2,
    pending_generics_count=3,
    first_to_file_exists=True,
    pte_applied=False,
    pediatric_exclusivity=False,
)

result = scorer.score_patent_cliff(drug_data)
print(f"Certainty Score: {result['scoring']['final_certainty_score']}%")
print(f"Recommendation: {result['trade_recommendation']['recommendation']}")
```

## Output Files

The pipeline generates several output files in the `output/` directory:

1. **CSV Calendar** (`patent_cliff_calendar_YYYYMMDD_HHMMSS.csv`)
   - Spreadsheet-friendly format
   - Contains all calendar events

2. **JSON Calendar** (`patent_cliff_calendar_YYYYMMDD_HHMMSS.json`)
   - Machine-readable format
   - Full event details

3. **Text Report** (`patent_cliff_report_YYYYMMDD_HHMMSS.txt`)
   - Human-readable summary
   - Organized by month

## Database Schema

### Core Tables
- `drugs`: Branded pharmaceutical products
- `patents`: Patent information with expiration dates
- `generic_applications`: ANDA filings and approvals
- `litigation`: Patent challenges and court cases
- `patent_cliff_calendar`: Upcoming events with scoring

### Views
- `v_upcoming_patent_expirations`: Patents expiring in next 18 months
- `v_generic_competition_summary`: Generic competition analysis
- `v_patent_cliff_calendar_12m`: 12-month calendar view

### Key Queries

Get upcoming high-value events:
```sql
SELECT * FROM v_patent_cliff_calendar_12m
WHERE certainty_score >= 80
  AND opportunity_tier IN ('BLOCKBUSTER', 'HIGH_VALUE')
ORDER BY event_date;
```

Get drugs with most generic competition:
```sql
SELECT * FROM v_generic_competition_summary
WHERE total_anda_filings >= 3
ORDER BY annual_revenue DESC;
```

## Scheduling

### Cron Job (Linux/Mac)
Add to crontab for weekly runs:
```bash
# Run every Monday at 8 AM
0 8 * * 1 cd /path/to/patent_intelligence && /path/to/venv/bin/python -m src.pipeline --send-email >> /var/log/patent_intelligence.log 2>&1
```

### Task Scheduler (Windows)
Create a scheduled task to run:
```
python -m src.pipeline --send-email
```

## Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
patent_intelligence/
├── config/
│   ├── config.yaml           # Main configuration
│   └── .env.template         # Environment variables template
├── sql/
│   └── schema.sql            # Database schema
├── src/
│   ├── __init__.py
│   ├── pipeline.py           # Main pipeline orchestrator
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── orange_book.py    # FDA Orange Book extractor
│   │   ├── uspto.py          # USPTO patent data extractor
│   │   └── fda_anda.py       # ANDA extractor
│   ├── transformers/
│   │   ├── __init__.py
│   │   ├── scoring.py        # Patent cliff scoring model
│   │   └── calendar.py       # Calendar generator
│   ├── loaders/
│   │   ├── __init__.py
│   │   └── db_loader.py      # Database loader
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Configuration management
│       ├── database.py       # Database connection
│       ├── email.py          # Email notifications
│       └── logger.py         # Logging setup
├── tests/
│   └── ...
├── output/                   # Generated reports
├── requirements.txt
└── README.md
```

## Troubleshooting

### Common Issues

**Database connection failed**:
- Verify PostgreSQL is running
- Check credentials in `.env`
- Ensure database exists

**Orange Book download failed**:
- Check internet connection
- FDA servers may be temporarily unavailable
- Try clearing cache: `rm -rf .cache/orange_book`

**No patents found for drug**:
- Some drugs may not have patent data in Orange Book
- Check NDA number format

**Email not sending**:
- Verify SMTP settings
- For Gmail, use app-specific password
- Check firewall settings

## Future Enhancements

Planned features for future versions:
- PACER litigation tracking
- AI patent claim analysis
- Real-time FDA approval alerts
- Web dashboard
- Historical backtesting

## License

Proprietary - Internal Use Only

## Support

For questions or issues, contact the development team.
