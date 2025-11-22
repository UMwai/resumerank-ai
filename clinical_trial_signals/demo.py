#!/usr/bin/env python3
"""
Clinical Trial Signal Detection System - Demonstration Script

This script demonstrates the system's capabilities by:
1. Fetching real Phase 3 trials from ClinicalTrials.gov
2. Fetching SEC 8-K filings for biotech companies
3. Showing how signals would be detected and scored
4. Generating a sample email digest

Run with: python demo.py

Note: This demo works without database dependencies by using standalone
versions of the scrapers. Full functionality requires PostgreSQL.
"""
import logging
import sys
import time
from datetime import datetime, date
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def print_header(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str):
    """Print a subsection header."""
    print(f"\n--- {title} ---\n")


# Standalone dataclass for demo (no database dependency)
@dataclass
class DemoTrialResult:
    """Parsed search result from ClinicalTrials.gov API."""
    nct_id: str
    title: str
    status: str
    phase: str
    sponsor: str
    conditions: List[str]
    interventions: List[str]
    enrollment: Optional[int]
    start_date: Optional[date]
    completion_date: Optional[date]


def demo_clinicaltrials_scraper():
    """Demonstrate the ClinicalTrials.gov scraper."""
    print_header("1. ClinicalTrials.gov Scraper Demo")

    print_subheader("Fetching Top 5 Phase 3 Oncology Trials")

    BASE_URL = "https://clinicaltrials.gov/api/v2"

    # Use simple query for Phase 3 trials
    params = {
        "query.term": "PHASE3",
        "filter.overallStatus": "RECRUITING",
        "pageSize": 5,
    }

    try:
        print("Connecting to ClinicalTrials.gov API...")
        response = requests.get(f"{BASE_URL}/studies", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        studies = data.get("studies", [])
        print(f"Found {len(studies)} trials\n")

        if not studies:
            print("No trials found (API may be temporarily unavailable)")
            return []

        trials = []
        for i, study in enumerate(studies, 1):
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status_module = protocol.get("statusModule", {})
            design_module = protocol.get("designModule", {})
            sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
            conditions_module = protocol.get("conditionsModule", {})

            nct_id = identification.get("nctId", "")
            title = identification.get("briefTitle", "")
            status = status_module.get("overallStatus", "")
            phases = design_module.get("phases", [])
            phase = phases[0] if phases else "Unknown"
            sponsor = sponsor_module.get("leadSponsor", {}).get("name", "Unknown")
            conditions = conditions_module.get("conditions", [])
            enrollment_info = design_module.get("enrollmentInfo", {})
            enrollment = enrollment_info.get("count")

            print(f"{i}. {nct_id}")
            print(f"   Title: {title[:60]}...")
            print(f"   Status: {status}")
            print(f"   Phase: {phase}")
            print(f"   Sponsor: {sponsor[:40]}...")
            print(f"   Conditions: {', '.join(conditions[:2])}")
            print(f"   Enrollment: {enrollment or 'N/A'}")
            print()

            trials.append(DemoTrialResult(
                nct_id=nct_id,
                title=title,
                status=status,
                phase=phase,
                sponsor=sponsor,
                conditions=conditions,
                interventions=[],
                enrollment=enrollment,
                start_date=None,
                completion_date=None
            ))

        return trials

    except requests.RequestException as e:
        print(f"Error fetching from ClinicalTrials.gov: {e}")
        print("Note: The API may be temporarily unavailable")
        return []


def demo_sec_scraper():
    """Demonstrate the SEC EDGAR scraper."""
    print_header("2. SEC EDGAR Scraper Demo")

    print_subheader("Looking up CIK for Moderna (MRNA)")

    BASE_URL = "https://data.sec.gov"
    USER_AGENT = "ClinicalTrialDemo demo@example.com"

    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    })

    try:
        # Get CIK - Use the correct SEC endpoint
        print("Fetching company tickers from SEC...")
        response = session.get("https://www.sec.gov/files/company_tickers.json", timeout=30)
        response.raise_for_status()
        data = response.json()

        cik = None
        for entry in data.values():
            if entry.get("ticker") == "MRNA":
                cik = str(entry.get("cik_str", "")).zfill(10)
                break

        if not cik:
            print("Could not find CIK for MRNA")
            return []

        print(f"Moderna CIK: {cik}")

        print_subheader("Fetching Recent 8-K Filings")

        time.sleep(0.2)  # Rate limiting

        # Get submissions
        response = session.get(f"{BASE_URL}/submissions/CIK{cik}.json", timeout=30)
        response.raise_for_status()
        data = response.json()

        company_name = data.get("name", "Unknown")
        print(f"Company: {company_name}\n")

        filings_data = data.get("filings", {}).get("recent", {})
        forms = filings_data.get("form", [])
        dates = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        descriptions = filings_data.get("primaryDocDescription", [])

        filings_shown = 0
        for i in range(min(len(forms), 50)):  # Check first 50
            if forms[i] == "8-K" and filings_shown < 5:
                filings_shown += 1
                print(f"{filings_shown}. {dates[i]}: 8-K")
                print(f"   Accession: {accessions[i]}")
                desc = descriptions[i] if i < len(descriptions) else ""
                print(f"   Description: {desc[:50]}..." if desc else "   (No description)")
                print()

        if filings_shown == 0:
            print("No recent 8-K filings found")

        return []

    except requests.RequestException as e:
        print(f"Error fetching from SEC: {e}")
        print("Note: SEC requires valid User-Agent header")
        return []


def demo_change_detection():
    """Demonstrate the change detection system."""
    print_header("3. Change Detection Demo")

    print_subheader("Simulating Trial Data Changes")

    # Create mock trial data (before)
    print("SCENARIO: Trial NCT99999999 has been updated")
    print("\nBEFORE (Previous Snapshot):")
    print("  Status: RECRUITING")
    print("  Enrollment: 450/500")
    print("  Expected Completion: 2025-12-01")
    print("  Primary Endpoint: Overall Survival at 24 months")

    print("\nAFTER (Current Data):")
    print("  Status: ACTIVE_NOT_RECRUITING")
    print("  Enrollment: 500/500 (FULL!)")
    print("  Expected Completion: 2025-09-01 (3 months earlier!)")
    print("  Primary Endpoint: Overall Survival at 24 months")

    print("\nDETECTED CHANGES:")
    changes = [
        {
            "type": "Status Change",
            "signal": "status_change_positive",
            "weight": "+2",
            "description": "Trial moved to active, recruitment complete"
        },
        {
            "type": "Enrollment",
            "signal": "early_enrollment",
            "weight": "+3",
            "description": "Enrollment target reached (500/500)"
        },
        {
            "type": "Completion Date",
            "signal": "completion_date_accelerated",
            "weight": "+3",
            "description": "Completion date moved earlier by 92 days"
        }
    ]

    for change in changes:
        print(f"\n  [{change['weight']}] {change['type']}")
        print(f"       Signal: {change['signal']}")
        print(f"       Description: {change['description']}")

    total_weight = sum(int(c["weight"].replace("+", "")) for c in changes)
    print(f"\n  TOTAL SIGNAL WEIGHT: +{total_weight}")


def demo_scoring():
    """Demonstrate the scoring model."""
    print_header("4. Signal Scoring Demo")

    # Signal weights (from config)
    weights = {
        "sites_added": 3,
        "early_enrollment": 3,
        "patent_filed": 5,
        "completion_date_accelerated": 3,
        "sec_8k_positive": 3,
        "status_change_positive": 2,
        "enrollment_increase": 2,
        "endpoint_change": -5,
        "completion_date_delayed": -3,
        "sec_8k_negative": -3,
        "enrollment_extended": -3,
        "status_change_negative": -2,
    }

    max_positive = sum(w for w in weights.values() if w > 0)
    max_negative = abs(sum(w for w in weights.values() if w < 0))

    print_subheader("Scoring Model Configuration")
    print(f"  Strong Buy Threshold: >= 7.0")
    print(f"  Buy Threshold: >= 5.0")
    print(f"  Short Threshold: <= 3.0")
    print(f"  Confidence Threshold: 0.7")
    print(f"  Max Positive Weight: {max_positive}")
    print(f"  Max Negative Weight: {max_negative}")

    print_subheader("Example Scoring Scenarios")

    scenarios = [
        {
            "name": "Strong Positive Trial",
            "signals": [
                ("status_change_positive", 2),
                ("early_enrollment", 3),
                ("completion_date_accelerated", 3),
                ("sec_8k_positive", 3),
            ],
            "expected": "STRONG_BUY"
        },
        {
            "name": "Mixed Signals Trial",
            "signals": [
                ("status_change_positive", 2),
                ("completion_date_delayed", -3),
                ("enrollment_increase", 2),
            ],
            "expected": "HOLD"
        },
        {
            "name": "Concerning Trial",
            "signals": [
                ("endpoint_change", -5),
                ("completion_date_delayed", -3),
                ("sec_8k_negative", -3),
            ],
            "expected": "SHORT"
        }
    ]

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print("  Signals:")

        total_weight = 0
        for signal_type, weight in scenario["signals"]:
            weight_str = f"+{weight}" if weight > 0 else str(weight)
            print(f"    [{weight_str}] {signal_type}")
            total_weight += weight

        # Calculate normalized score
        if total_weight >= 0:
            score = 5 + (total_weight / max_positive * 5)
        else:
            score = 5 + (total_weight / max_negative * 5)
        score = max(0, min(10, score))

        # Confidence based on signal count
        signal_count = len(scenario["signals"])
        confidence = min(signal_count / 6.0, 1.0) * 0.5 + 0.3

        # Determine recommendation
        if score >= 7.0 and confidence >= 0.7:
            rec = "STRONG_BUY"
        elif score >= 5.0:
            rec = "BUY"
        elif score <= 3.0 and confidence >= 0.7:
            rec = "SHORT"
        else:
            rec = "HOLD"

        print(f"\n  RESULT:")
        print(f"    Composite Score: {score:.1f}/10")
        print(f"    Confidence: {confidence:.0%}")
        print(f"    Recommendation: {rec}")


def demo_email_digest():
    """Demonstrate email digest generation."""
    print_header("5. Email Digest Demo")

    print_subheader("Sample Daily Digest")

    digest_text = """
============================================================
CLINICAL TRIAL SIGNAL REPORT
November 22, 2025
============================================================

SUMMARY
----------------------------------------
Monitoring: 20 trials
With active signals: 8
New signals today: 5

TOP BUY OPPORTUNITIES
----------------------------------------
1. Experimental Drug X (ABCD)
   Score: 8.2/10 | STRONG_BUY
   Trial: NCT12345678
   Indication: Non-Small Cell Lung Cancer

2. Novel Therapy Y (EFGH)
   Score: 7.5/10 | BUY
   Trial: NCT23456789
   Indication: Breast Cancer

SHORT OPPORTUNITIES
----------------------------------------
1. Treatment Z (IJKL)
   Score: 2.3/10 | SHORT
   Trial: NCT34567890
   Indication: Multiple Myeloma

RECENT SIGNALS
----------------------------------------
[+3] NCT12345678 - early_enrollment: Enrollment target reached
[+3] NCT12345678 - completion_date_accelerated: 3 months earlier
[+2] NCT23456789 - status_change_positive: Moved to active phase
[-5] NCT34567890 - endpoint_change: Primary endpoint modified
[-3] NCT34567890 - completion_date_delayed: Delayed by 6 months

============================================================
DISCLAIMER: This is not financial advice.
Generated: 2025-11-22 08:00:00 UTC
"""
    print(digest_text)


def demo_full_pipeline():
    """Show full pipeline output example."""
    print_header("6. Full Pipeline Execution Example")

    print("""
$ python main.py --full

2025-11-22 08:00:00 | INFO     | Starting Clinical Trial Signal Detection Pipeline
2025-11-22 08:00:00 | INFO     | Timestamp: 2025-11-22T08:00:00
2025-11-22 08:00:00 | INFO     | Dry run: False
============================================================

2025-11-22 08:00:01 | INFO     | Fetching top 20 Phase 3 trials from ClinicalTrials.gov...
2025-11-22 08:00:05 | INFO     | Found 20 Phase 3 trials
2025-11-22 08:00:06 | INFO     | Stored 3 new trials, updated 17 existing trials

2025-11-22 08:00:07 | INFO     | Fetching SEC 8-K filings from last 7 days...
2025-11-22 08:00:15 | INFO     | Processed 12 filings, found 3 signals

2025-11-22 08:00:16 | INFO     | Running change detection on monitored trials...
2025-11-22 08:00:18 | INFO     | Checked 20 trials, detected 4 with changes, created 8 signals

2025-11-22 08:00:19 | INFO     | Calculating composite scores...
2025-11-22 08:00:20 | INFO     | Scored 8 trials: 2 buys, 1 shorts

2025-11-22 08:00:21 | INFO     | Generating email digest...
2025-11-22 08:00:22 | INFO     | Email sent successfully to 2 recipients

============================================================
PIPELINE EXECUTION SUMMARY
============================================================

FETCH_TRIALS:
  new_trials: 3
  updated_trials: 17

FETCH_SEC:
  filings_processed: 12
  signals_found: 3

CHANGE_DETECTION:
  trials_checked: 20
  changes_detected: 4
  signals_created: 8

SCORING:
  trials_scored: 8
  strong_buys: 2
  shorts: 1

EMAIL:
  sent: True
  recipients: 2
  signals_included: 11

Completed at: 2025-11-22T08:00:22
============================================================

2025-11-22 08:00:22 | INFO     | Pipeline completed successfully
""")


def main():
    """Run all demonstrations."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#    CLINICAL TRIAL SIGNAL DETECTION SYSTEM - DEMONSTRATION" + " " * 9 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    print("\nThis demo shows the system's capabilities using real API data where")
    print("available, and simulated examples for database-dependent features.")
    print("\nNote: No database connection required for this demo.")

    try:
        # Demo each component
        demo_clinicaltrials_scraper()
        demo_sec_scraper()
        demo_change_detection()
        demo_scoring()
        demo_email_digest()
        demo_full_pipeline()

        print_header("DEMONSTRATION COMPLETE")
        print("""
The Clinical Trial Signal Detection System is ready for deployment!

Next steps:
1. Install dependencies: pip install -r requirements.txt
2. Copy config_template.env to .env and configure
3. Initialize PostgreSQL database: python main.py --init-db
4. Run a dry-run test: python main.py --dry-run --full
5. Run the full pipeline: python main.py --full
6. Set up daily cron job for automated monitoring

For more information, see README.md
""")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        logger.error(f"Demo error: {e}")
        print(f"\nError during demo: {e}")
        print("Some features require network access to ClinicalTrials.gov and SEC APIs")


if __name__ == "__main__":
    main()
