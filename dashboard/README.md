# Investment Intelligence Dashboard

A Streamlit-based dashboard that integrates data from three intelligence systems to provide unified investment signals for biotech and pharmaceutical companies.

## Overview

This dashboard combines signals from:

1. **Clinical Trial Signal Detection** - Monitors biotech clinical trials for investment-relevant signals
2. **Patent/IP Intelligence** - Tracks patent expirations and generic entry opportunities
3. **Insider Activity + Hiring Signals** - Analyzes insider transactions, institutional holdings, and hiring patterns

## Features

### Home/Overview Page
- Combined opportunities from all 3 systems, scored by confidence
- Key metrics overview
- Active alerts summary
- Top opportunities with score breakdown

### Clinical Trials Tab
- Recent trial signals
- Upcoming readouts calendar
- Top opportunities (bullish and bearish)
- All monitored trials

### Patent Cliff Tab
- 12-month patent expiration calendar
- Generic entry opportunities
- Trade recommendations
- Litigation status tracking

### Insider/Hiring Tab
- Recent Form 4 filings
- 13F institutional holding changes
- Job posting trends
- Composite signal scores

### Watchlist
- Add/remove stocks to monitor
- Combined signals for watchlist items
- Individual stock details

### Alerts
- Configurable alert thresholds
- High-confidence signal notifications
- Alert history

## Installation

### Prerequisites

- Python 3.9 or higher
- PostgreSQL databases (optional - can run in demo mode)

### Setup

1. **Clone the repository** (if not already done):
   ```bash
   cd /path/to/dreamers-v2
   ```

2. **Create a virtual environment**:
   ```bash
   cd dashboard
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the database connections**:
   ```bash
   cp config.yaml.template config.yaml
   # Edit config.yaml with your database credentials
   ```

5. **Run the dashboard**:
   ```bash
   streamlit run app.py
   ```

6. **Open in browser**:
   The dashboard will open automatically at `http://localhost:8501`

## Configuration

### config.yaml

Copy `config.yaml.template` to `config.yaml` and configure:

```yaml
# Set to false to connect to actual databases
demo_mode: false

databases:
  clinical_trials:
    host: localhost
    port: 5432
    database: clinical_trials
    user: postgres
    password: your_password

  patent_intelligence:
    host: localhost
    port: 5432
    database: patent_intelligence
    user: postgres
    password: your_password

  insider_hiring:
    host: localhost
    port: 5432
    database: insider_signals
    user: postgres
    password: your_password
```

### Environment Variables

You can also configure database connections using environment variables:

```bash
# Clinical Trials Database
export CT_DB_HOST=localhost
export CT_DB_PORT=5432
export CT_DB_NAME=clinical_trials
export CT_DB_USER=postgres
export CT_DB_PASSWORD=your_password

# Patent Intelligence Database
export PI_DB_HOST=localhost
export PI_DB_PORT=5432
export PI_DB_NAME=patent_intelligence
export PI_DB_USER=postgres
export PI_DB_PASSWORD=your_password

# Insider Hiring Database
export IH_DB_HOST=localhost
export IH_DB_PORT=5432
export IH_DB_NAME=insider_signals
export IH_DB_USER=postgres
export IH_DB_PASSWORD=your_password
```

## Project Structure

```
dashboard/
├── app.py                  # Main Streamlit application
├── config.yaml.template    # Configuration template
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── components/            # Reusable UI components
│   ├── __init__.py
│   ├── cards.py          # Card components (metrics, signals, alerts)
│   ├── charts.py         # Plotly chart components
│   ├── filters.py        # Filter components
│   └── tables.py         # Table components
├── pages/                 # Page modules
│   ├── __init__.py
│   ├── home.py           # Home/Overview page
│   ├── clinical_trials.py # Clinical Trials page
│   ├── patent_cliff.py   # Patent Cliff page
│   ├── insider_hiring.py # Insider/Hiring page
│   ├── watchlist.py      # Watchlist page
│   └── alerts.py         # Alerts page
└── utils/                 # Utility modules
    ├── __init__.py
    ├── database.py       # Database connection manager
    └── data_fetchers.py  # Data fetching classes
```

## Combined Score Calculation

The dashboard calculates a combined confidence score across all systems:

```python
combined_score = (
    clinical_trial_score * 0.4 +   # Highest weight - binary events
    patent_cliff_score * 0.3 +      # Medium weight - predictable events
    insider_hiring_score * 0.3      # Medium weight - directional signals
)
```

### Recommendation Thresholds

| Score Range | Confidence | Recommendation |
|-------------|------------|----------------|
| >= 0.7      | >= 0.7     | STRONG BUY     |
| >= 0.6      | >= 0.5     | BUY            |
| 0.4 - 0.6   | Any        | HOLD           |
| <= 0.4      | >= 0.5     | SELL           |
| <= 0.3      | >= 0.7     | STRONG SELL    |

## UI/UX Guidelines

### Color Coding
- **Green**: Bullish signals (BUY, STRONG BUY)
- **Red**: Bearish signals (SELL, STRONG SELL)
- **Yellow/Orange**: Neutral or warning signals
- **Gray**: No signal or inactive

### Tables
- All tables are sortable by clicking column headers
- Filter controls are provided for most data views
- Pagination is available for large datasets

### Responsiveness
- The dashboard is designed to work on various screen sizes
- Sidebar can be collapsed on smaller screens

## Data Refresh

- Click the "Refresh Data" button in the sidebar to update all data
- Individual pages have refresh buttons for page-specific data
- Data is cached to improve performance (cache clears on refresh)

## Development

### Running in Demo Mode

The dashboard can run without database connections using demo mode:

```yaml
# config.yaml
demo_mode: true
```

This displays sample data for testing and development.

### Adding New Pages

1. Create a new file in `pages/`:
   ```python
   # pages/my_page.py
   def render_my_page(fetcher):
       st.title("My Page")
       # ... page content
   ```

2. Import and add to navigation in `app.py`

### Creating Custom Components

Add new components in the `components/` directory and export them in `__init__.py`.

## Troubleshooting

### Connection Issues

1. Verify database credentials in `config.yaml`
2. Ensure PostgreSQL is running and accessible
3. Check network/firewall settings

### Performance Issues

1. Reduce the number of items displayed per page
2. Clear the cache using the refresh button
3. Check database query performance

### Missing Data

1. Verify the data pipelines for each system are running
2. Check database tables have been populated
3. Ensure the correct database is being connected to

## License

Part of the Dreamers-v2 project. See the main repository for license information.
