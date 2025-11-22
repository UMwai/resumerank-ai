"""
Real-time FDA/PACER Monitoring

Monitors FDA approvals and PACER litigation updates with Slack alerting.
Includes budget tracking to stay under $30/month PACER limit.

Features:
- RSS feed monitoring for FDA approvals
- Daily PACER checks for litigation updates
- Slack alerts for critical events
- Budget tracking for PACER costs
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from urllib.parse import urlencode

import requests

from ..utils.logger import get_logger

logger = get_logger(__name__)

# Try to import optional dependencies
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger.warning("feedparser not installed. Install with: pip install feedparser")

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    logger.warning("slack_sdk not installed. Install with: pip install slack-sdk")


@dataclass
class MonitorEvent:
    """Event detected by monitoring system."""

    event_type: str  # FDA_APPROVAL, ANDA_APPROVAL, LITIGATION_FILED, LITIGATION_UPDATE
    drug_name: str
    company: Optional[str]
    description: str
    source: str  # FDA, PACER
    source_url: Optional[str]
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    detected_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["detected_at"] = self.detected_at.isoformat()
        return result

    def to_slack_message(self) -> Dict[str, Any]:
        """Format as Slack message block."""
        # Severity colors
        colors = {
            "LOW": "#36a64f",
            "MEDIUM": "#ffcc00",
            "HIGH": "#ff6600",
            "CRITICAL": "#ff0000",
        }

        return {
            "attachments": [
                {
                    "color": colors.get(self.severity, "#808080"),
                    "blocks": [
                        {
                            "type": "header",
                            "text": {
                                "type": "plain_text",
                                "text": f"{self.severity}: {self.event_type}",
                            },
                        },
                        {
                            "type": "section",
                            "fields": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Drug:*\n{self.drug_name}",
                                },
                                {
                                    "type": "mrkdwn",
                                    "text": f"*Company:*\n{self.company or 'N/A'}",
                                },
                            ],
                        },
                        {
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Description:*\n{self.description}",
                            },
                        },
                        {
                            "type": "context",
                            "elements": [
                                {
                                    "type": "mrkdwn",
                                    "text": f"Source: {self.source} | {self.detected_at.strftime('%Y-%m-%d %H:%M')}",
                                },
                            ],
                        },
                    ],
                }
            ]
        }


class BudgetTracker:
    """
    Tracks PACER spending to stay under monthly budget.
    """

    def __init__(
        self,
        monthly_budget: float = 30.0,
        cost_per_page: float = 0.10,
        state_file: Optional[str] = None,
    ):
        """
        Initialize budget tracker.

        Args:
            monthly_budget: Monthly budget in dollars.
            cost_per_page: Cost per PACER page.
            state_file: Path to state file for persistence.
        """
        self.monthly_budget = monthly_budget
        self.cost_per_page = cost_per_page
        self.state_file = state_file or str(
            Path(__file__).parent.parent.parent / "output" / "pacer_budget.json"
        )

        # Load or initialize state
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        """Load state from file."""
        if Path(self.state_file).exists():
            try:
                with open(self.state_file, "r") as f:
                    state = json.load(f)

                # Reset if new month
                if state.get("month") != date.today().strftime("%Y-%m"):
                    return self._new_month_state()

                return state
            except Exception as e:
                logger.error(f"Error loading budget state: {e}")

        return self._new_month_state()

    def _new_month_state(self) -> Dict[str, Any]:
        """Create new month state."""
        return {
            "month": date.today().strftime("%Y-%m"),
            "total_spent": 0.0,
            "total_pages": 0,
            "queries": [],
            "last_query_date": None,
        }

    def _save_state(self) -> None:
        """Save state to file."""
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving budget state: {e}")

    def can_query(self, estimated_pages: int = 10) -> bool:
        """
        Check if we can make a query within budget.

        Args:
            estimated_pages: Estimated number of pages.

        Returns:
            True if within budget.
        """
        estimated_cost = estimated_pages * self.cost_per_page
        return (self.state["total_spent"] + estimated_cost) <= self.monthly_budget

    def record_query(self, pages: int, description: str) -> float:
        """
        Record a PACER query.

        Args:
            pages: Number of pages retrieved.
            description: Query description.

        Returns:
            Cost of this query.
        """
        cost = pages * self.cost_per_page

        self.state["total_spent"] += cost
        self.state["total_pages"] += pages
        self.state["last_query_date"] = datetime.now().isoformat()
        self.state["queries"].append({
            "date": datetime.now().isoformat(),
            "pages": pages,
            "cost": cost,
            "description": description,
        })

        self._save_state()

        logger.info(
            f"PACER query: {pages} pages, ${cost:.2f}. "
            f"Month total: ${self.state['total_spent']:.2f}/{self.monthly_budget:.2f}"
        )

        return cost

    def get_remaining_budget(self) -> float:
        """Get remaining budget for the month."""
        return max(0, self.monthly_budget - self.state["total_spent"])

    def get_status(self) -> Dict[str, Any]:
        """Get budget status."""
        return {
            "month": self.state["month"],
            "budget": self.monthly_budget,
            "spent": round(self.state["total_spent"], 2),
            "remaining": round(self.get_remaining_budget(), 2),
            "pages_retrieved": self.state["total_pages"],
            "query_count": len(self.state["queries"]),
            "last_query": self.state["last_query_date"],
            "percent_used": round(
                (self.state["total_spent"] / self.monthly_budget) * 100, 1
            ),
        }


class FDAMonitor:
    """
    Monitors FDA for drug approvals and ANDA updates.
    """

    # FDA RSS feeds
    FDA_FEEDS = {
        "drug_approvals": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-approvals/rss.xml",
        "drug_safety": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/drug-safety-communications/rss.xml",
        "press_releases": "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/press-releases/rss.xml",
    }

    # FDA API endpoints
    FDA_API_BASE = "https://api.fda.gov/drug"

    def __init__(
        self,
        watched_drugs: Optional[List[str]] = None,
        watched_companies: Optional[List[str]] = None,
        state_file: Optional[str] = None,
    ):
        """
        Initialize FDA monitor.

        Args:
            watched_drugs: List of drug names to watch.
            watched_companies: List of company names to watch.
            state_file: Path to state file.
        """
        self.watched_drugs = [d.lower() for d in (watched_drugs or [])]
        self.watched_companies = [c.lower() for c in (watched_companies or [])]
        self.state_file = state_file or str(
            Path(__file__).parent.parent.parent / "output" / "fda_monitor_state.json"
        )

        self.seen_entries = self._load_seen_entries()

    def _load_seen_entries(self) -> set:
        """Load previously seen entries."""
        if Path(self.state_file).exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                return set(data.get("seen_hashes", []))
            except Exception:
                pass
        return set()

    def _save_seen_entries(self) -> None:
        """Save seen entries."""
        try:
            Path(self.state_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump({
                    "seen_hashes": list(self.seen_entries),
                    "last_check": datetime.now().isoformat(),
                }, f)
        except Exception as e:
            logger.error(f"Error saving FDA monitor state: {e}")

    def _entry_hash(self, entry: Dict[str, Any]) -> str:
        """Generate hash for entry to detect duplicates."""
        content = f"{entry.get('title', '')}{entry.get('link', '')}"
        return hashlib.md5(content.encode()).hexdigest()

    def _is_relevant(self, text: str) -> bool:
        """Check if text is relevant to watched items."""
        text_lower = text.lower()

        # Check drugs
        for drug in self.watched_drugs:
            if drug in text_lower:
                return True

        # Check companies
        for company in self.watched_companies:
            if company in text_lower:
                return True

        # Check for generic/ANDA keywords
        anda_keywords = ["anda", "generic", "biosimilar", "abbreviated new drug"]
        for keyword in anda_keywords:
            if keyword in text_lower:
                return True

        return False

    def check_feeds(self) -> List[MonitorEvent]:
        """
        Check FDA RSS feeds for new entries.

        Returns:
            List of new relevant events.
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available, skipping RSS check")
            return []

        events = []

        for feed_name, feed_url in self.FDA_FEEDS.items():
            try:
                logger.info(f"Checking FDA feed: {feed_name}")
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    entry_hash = self._entry_hash(entry)

                    # Skip if already seen
                    if entry_hash in self.seen_entries:
                        continue

                    self.seen_entries.add(entry_hash)

                    # Check relevance
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    combined = f"{title} {summary}"

                    if self._is_relevant(combined):
                        # Determine event type and severity
                        event_type, severity = self._classify_entry(title, summary)

                        event = MonitorEvent(
                            event_type=event_type,
                            drug_name=self._extract_drug_name(combined),
                            company=self._extract_company(combined),
                            description=summary[:500] if summary else title,
                            source="FDA",
                            source_url=entry.get("link"),
                            severity=severity,
                            metadata={
                                "feed": feed_name,
                                "published": entry.get("published"),
                            },
                        )
                        events.append(event)

                        logger.info(f"New FDA event: {event_type} - {title[:50]}")

            except Exception as e:
                logger.error(f"Error checking FDA feed {feed_name}: {e}")

        self._save_seen_entries()
        return events

    def _classify_entry(self, title: str, summary: str) -> tuple:
        """Classify entry type and severity."""
        combined = f"{title} {summary}".lower()

        if "anda" in combined or "generic" in combined:
            if "approv" in combined:
                return "ANDA_APPROVAL", "HIGH"
            elif "fil" in combined or "submit" in combined:
                return "ANDA_FILING", "MEDIUM"
            else:
                return "ANDA_UPDATE", "LOW"

        if "biosimilar" in combined:
            if "approv" in combined:
                return "BIOSIMILAR_APPROVAL", "HIGH"
            else:
                return "BIOSIMILAR_UPDATE", "MEDIUM"

        if "approv" in combined:
            return "FDA_APPROVAL", "MEDIUM"

        if "warning" in combined or "safety" in combined:
            return "SAFETY_ALERT", "HIGH"

        return "FDA_UPDATE", "LOW"

    def _extract_drug_name(self, text: str) -> str:
        """Extract drug name from text."""
        text_lower = text.lower()

        for drug in self.watched_drugs:
            if drug in text_lower:
                return drug.title()

        return "Unknown"

    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name from text."""
        text_lower = text.lower()

        for company in self.watched_companies:
            if company in text_lower:
                return company.title()

        return None

    def check_anda_approvals(self, days_back: int = 7) -> List[MonitorEvent]:
        """
        Check FDA API for recent ANDA approvals.

        Args:
            days_back: Number of days to look back.

        Returns:
            List of ANDA approval events.
        """
        events = []

        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)

        # Query FDA drug approvals API
        params = {
            "search": f'submissions.submission_type:"ANDA" AND submissions.submission_status_date:[{start_date.strftime("%Y%m%d")} TO {end_date.strftime("%Y%m%d")}]',
            "limit": 100,
        }

        url = f"{self.FDA_API_BASE}/drugsfda.json?{urlencode(params)}"

        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json()

                for result in data.get("results", []):
                    brand_name = result.get("openfda", {}).get("brand_name", ["Unknown"])[0]
                    generic_name = result.get("openfda", {}).get("generic_name", ["Unknown"])[0]
                    sponsor = result.get("sponsor_name", "Unknown")

                    # Check relevance
                    if not self._is_relevant(f"{brand_name} {generic_name} {sponsor}"):
                        continue

                    event = MonitorEvent(
                        event_type="ANDA_APPROVAL",
                        drug_name=f"{generic_name} ({brand_name})",
                        company=sponsor,
                        description=f"ANDA approved for {generic_name}",
                        source="FDA_API",
                        source_url=None,
                        severity="HIGH",
                        metadata={
                            "application_number": result.get("application_number"),
                            "submissions": result.get("submissions", []),
                        },
                    )
                    events.append(event)

        except Exception as e:
            logger.error(f"Error checking FDA API: {e}")

        return events


class PACERMonitor:
    """
    Monitors PACER for litigation updates.

    Note: Actual PACER integration requires an account and paid access.
    This implementation provides the framework with mock data for testing.
    """

    # PACER costs and limits
    COST_PER_PAGE = 0.10
    PAGES_PER_SEARCH = 10  # Estimated pages per search

    # Common courts for pharma patent cases
    PRIORITY_COURTS = [
        "ded",  # District of Delaware
        "njd",  # District of New Jersey
        "cand",  # Northern District of California
        "txed",  # Eastern District of Texas
    ]

    def __init__(
        self,
        watched_cases: Optional[List[str]] = None,
        watched_patents: Optional[List[str]] = None,
        budget_tracker: Optional[BudgetTracker] = None,
    ):
        """
        Initialize PACER monitor.

        Args:
            watched_cases: List of case numbers to watch.
            watched_patents: List of patent numbers to watch.
            budget_tracker: Budget tracker instance.
        """
        self.watched_cases = watched_cases or []
        self.watched_patents = watched_patents or []
        self.budget_tracker = budget_tracker or BudgetTracker()

        # Mock data for testing (would be replaced with real PACER API)
        self._mock_cases = self._init_mock_data()

    def _init_mock_data(self) -> List[Dict[str, Any]]:
        """Initialize mock case data for testing."""
        return [
            {
                "case_number": "1:23-cv-01234",
                "court": "ded",
                "title": "Teva Pharm. v. AbbVie Inc.",
                "drug": "Humira",
                "patent": "6090382",
                "filing_date": "2023-06-15",
                "status": "ACTIVE",
                "last_update": datetime.now().isoformat(),
                "recent_filings": [
                    {
                        "date": (datetime.now() - timedelta(days=2)).isoformat(),
                        "type": "MOTION",
                        "description": "Motion for Summary Judgment",
                    },
                ],
            },
            {
                "case_number": "1:22-cv-05678",
                "court": "njd",
                "title": "Bristol-Myers Squibb v. Unichem Labs",
                "drug": "Eliquis",
                "patent": "7371746",
                "filing_date": "2022-09-20",
                "status": "ACTIVE",
                "last_update": datetime.now().isoformat(),
                "recent_filings": [
                    {
                        "date": (datetime.now() - timedelta(days=5)).isoformat(),
                        "type": "ORDER",
                        "description": "Court Order Setting Trial Date",
                    },
                ],
            },
            {
                "case_number": "1:21-cv-09999",
                "court": "ded",
                "title": "Merck v. Generic Challenger",
                "drug": "Keytruda",
                "patent": "8354509",
                "filing_date": "2021-11-10",
                "status": "SETTLED",
                "last_update": (datetime.now() - timedelta(days=30)).isoformat(),
                "recent_filings": [],
            },
        ]

    def check_cases(self) -> List[MonitorEvent]:
        """
        Check watched cases for updates.

        Returns:
            List of litigation events.
        """
        events = []

        # Check budget
        if not self.budget_tracker.can_query(self.PAGES_PER_SEARCH):
            logger.warning("PACER budget exceeded, skipping check")
            return []

        # In production, this would query PACER
        # For now, use mock data
        for case in self._mock_cases:
            if case["status"] != "ACTIVE":
                continue

            # Check for recent filings
            for filing in case.get("recent_filings", []):
                filing_date = datetime.fromisoformat(filing["date"])

                # Only report filings from last 7 days
                if filing_date > datetime.now() - timedelta(days=7):
                    severity = self._classify_filing_severity(filing["type"])

                    event = MonitorEvent(
                        event_type="LITIGATION_UPDATE",
                        drug_name=case["drug"],
                        company=case["title"].split(" v. ")[0] if " v. " in case["title"] else None,
                        description=f"{filing['type']}: {filing['description']}",
                        source="PACER",
                        source_url=f"https://pacer.uscourts.gov/case/{case['case_number']}",
                        severity=severity,
                        metadata={
                            "case_number": case["case_number"],
                            "court": case["court"],
                            "patent": case["patent"],
                            "filing_type": filing["type"],
                        },
                    )
                    events.append(event)

        # Record budget usage (mock)
        if events:
            self.budget_tracker.record_query(
                pages=self.PAGES_PER_SEARCH,
                description=f"Case check: {len(events)} updates found"
            )

        return events

    def _classify_filing_severity(self, filing_type: str) -> str:
        """Classify severity based on filing type."""
        high_severity = ["ORDER", "JUDGMENT", "VERDICT", "INJUNCTION"]
        medium_severity = ["MOTION", "BRIEF", "OPINION"]

        filing_upper = filing_type.upper()

        if any(s in filing_upper for s in high_severity):
            return "HIGH"
        elif any(s in filing_upper for s in medium_severity):
            return "MEDIUM"
        else:
            return "LOW"

    def search_new_cases(
        self,
        drug_name: Optional[str] = None,
        patent_number: Optional[str] = None,
    ) -> List[MonitorEvent]:
        """
        Search for new litigation cases.

        Args:
            drug_name: Drug name to search.
            patent_number: Patent number to search.

        Returns:
            List of new case events.
        """
        events = []

        # Check budget
        if not self.budget_tracker.can_query(self.PAGES_PER_SEARCH * 2):
            logger.warning("PACER budget exceeded, skipping search")
            return []

        # Mock implementation - in production would query PACER
        logger.info(f"Searching PACER for: drug={drug_name}, patent={patent_number}")

        # Record budget usage
        self.budget_tracker.record_query(
            pages=self.PAGES_PER_SEARCH,
            description=f"New case search: {drug_name or patent_number}"
        )

        return events

    def get_budget_status(self) -> Dict[str, Any]:
        """Get current PACER budget status."""
        return self.budget_tracker.get_status()


class SlackNotifier:
    """
    Sends notifications to Slack.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        channel: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        """
        Initialize Slack notifier.

        Args:
            token: Slack bot token.
            channel: Default channel to post to.
            webhook_url: Webhook URL for simple posting.
        """
        self.token = token or os.getenv("SLACK_BOT_TOKEN")
        self.channel = channel or os.getenv("SLACK_CHANNEL", "#patent-alerts")
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")

        self.client = None
        if SLACK_AVAILABLE and self.token:
            self.client = WebClient(token=self.token)

    def send_event(
        self,
        event: MonitorEvent,
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send event notification to Slack.

        Args:
            event: Event to notify about.
            channel: Override channel.

        Returns:
            True if sent successfully.
        """
        target_channel = channel or self.channel

        # Try webhook first (simpler, no SDK needed)
        if self.webhook_url:
            return self._send_webhook(event)

        # Try SDK client
        if self.client:
            return self._send_client(event, target_channel)

        logger.warning("No Slack configuration available")
        return False

    def _send_webhook(self, event: MonitorEvent) -> bool:
        """Send via webhook."""
        try:
            message = event.to_slack_message()
            response = requests.post(
                self.webhook_url,
                json=message,
                timeout=10,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Slack webhook error: {e}")
            return False

    def _send_client(self, event: MonitorEvent, channel: str) -> bool:
        """Send via SDK client."""
        if not SLACK_AVAILABLE:
            return False

        try:
            message = event.to_slack_message()
            self.client.chat_postMessage(
                channel=channel,
                attachments=message["attachments"],
            )
            return True
        except SlackApiError as e:
            logger.error(f"Slack API error: {e.response['error']}")
            return False

    def send_summary(
        self,
        events: List[MonitorEvent],
        channel: Optional[str] = None,
    ) -> bool:
        """
        Send summary of multiple events.

        Args:
            events: List of events.
            channel: Override channel.

        Returns:
            True if sent successfully.
        """
        if not events:
            return True

        # Group by severity
        by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": []}
        for event in events:
            by_severity[event.severity].append(event)

        # Build summary text
        lines = [f"*Patent Intelligence Alert Summary* ({len(events)} events)"]

        for severity, severity_events in by_severity.items():
            if severity_events:
                lines.append(f"\n*{severity}* ({len(severity_events)}):")
                for event in severity_events[:5]:  # Limit per category
                    lines.append(f"  - {event.event_type}: {event.drug_name}")

        summary_text = "\n".join(lines)

        # Send via webhook or client
        if self.webhook_url:
            try:
                response = requests.post(
                    self.webhook_url,
                    json={"text": summary_text},
                    timeout=10,
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Slack webhook error: {e}")
                return False

        if self.client and SLACK_AVAILABLE:
            try:
                self.client.chat_postMessage(
                    channel=channel or self.channel,
                    text=summary_text,
                )
                return True
            except Exception as e:
                logger.error(f"Slack client error: {e}")
                return False

        return False


class RealtimeMonitor:
    """
    Main real-time monitoring coordinator.

    Combines FDA and PACER monitoring with Slack alerting.
    """

    def __init__(
        self,
        watched_drugs: Optional[List[str]] = None,
        watched_companies: Optional[List[str]] = None,
        watched_patents: Optional[List[str]] = None,
        pacer_budget: float = 30.0,
        slack_channel: Optional[str] = None,
    ):
        """
        Initialize real-time monitor.

        Args:
            watched_drugs: Drugs to monitor.
            watched_companies: Companies to monitor.
            watched_patents: Patents to monitor.
            pacer_budget: Monthly PACER budget.
            slack_channel: Slack channel for alerts.
        """
        # Initialize budget tracker
        self.budget_tracker = BudgetTracker(monthly_budget=pacer_budget)

        # Initialize monitors
        self.fda_monitor = FDAMonitor(
            watched_drugs=watched_drugs,
            watched_companies=watched_companies,
        )
        self.pacer_monitor = PACERMonitor(
            watched_patents=watched_patents,
            budget_tracker=self.budget_tracker,
        )

        # Initialize notifier
        self.slack = SlackNotifier(channel=slack_channel)

        # Event handlers
        self.event_handlers: List[Callable[[MonitorEvent], None]] = []

    def add_handler(self, handler: Callable[[MonitorEvent], None]) -> None:
        """Add custom event handler."""
        self.event_handlers.append(handler)

    def check_all(self) -> List[MonitorEvent]:
        """
        Run all monitoring checks.

        Returns:
            List of detected events.
        """
        events = []

        # Check FDA
        logger.info("Running FDA checks...")
        fda_events = self.fda_monitor.check_feeds()
        events.extend(fda_events)

        # Check PACER
        logger.info("Running PACER checks...")
        pacer_events = self.pacer_monitor.check_cases()
        events.extend(pacer_events)

        # Process events
        for event in events:
            # Call handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Handler error: {e}")

            # Send to Slack for HIGH/CRITICAL
            if event.severity in ["HIGH", "CRITICAL"]:
                self.slack.send_event(event)

        # Send summary if any events
        if events:
            self.slack.send_summary(events)

        logger.info(f"Monitoring complete: {len(events)} events detected")
        return events

    def get_status(self) -> Dict[str, Any]:
        """Get monitoring system status."""
        return {
            "pacer_budget": self.budget_tracker.get_status(),
            "fda_feeds": list(FDAMonitor.FDA_FEEDS.keys()),
            "slack_configured": bool(self.slack.webhook_url or self.slack.token),
            "feedparser_available": FEEDPARSER_AVAILABLE,
            "slack_sdk_available": SLACK_AVAILABLE,
        }


# Default watched items for pharmaceutical patent monitoring
DEFAULT_WATCHED_DRUGS = [
    "humira", "adalimumab",
    "eliquis", "apixaban",
    "keytruda", "pembrolizumab",
    "revlimid", "lenalidomide",
    "stelara", "ustekinumab",
    "eylea", "aflibercept",
    "xarelto", "rivaroxaban",
    "ibrance", "palbociclib",
    "opdivo", "nivolumab",
    "entresto", "sacubitril",
]

DEFAULT_WATCHED_COMPANIES = [
    "abbvie",
    "bristol-myers squibb",
    "merck",
    "pfizer",
    "johnson & johnson",
    "roche",
    "novartis",
    "amgen",
    "teva",
    "mylan",
]


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("Real-time FDA/PACER Monitoring System")
    print("=" * 70)

    # Initialize monitor
    monitor = RealtimeMonitor(
        watched_drugs=DEFAULT_WATCHED_DRUGS[:5],
        watched_companies=DEFAULT_WATCHED_COMPANIES[:5],
        pacer_budget=30.0,
    )

    # Get status
    status = monitor.get_status()
    print("\nSystem Status:")
    print(f"  PACER Budget: ${status['pacer_budget']['remaining']:.2f} remaining")
    print(f"  FDA Feeds: {len(status['fda_feeds'])} configured")
    print(f"  Slack Configured: {status['slack_configured']}")
    print(f"  feedparser Available: {status['feedparser_available']}")
    print(f"  slack_sdk Available: {status['slack_sdk_available']}")

    # Run checks
    print("\nRunning monitoring checks...")
    events = monitor.check_all()

    print(f"\nEvents Detected: {len(events)}")
    for event in events:
        print(f"\n  {event.severity}: {event.event_type}")
        print(f"    Drug: {event.drug_name}")
        print(f"    Source: {event.source}")
        print(f"    Description: {event.description[:100]}...")

    # Show budget status
    budget = monitor.budget_tracker.get_status()
    print("\n" + "=" * 70)
    print("PACER Budget Status")
    print("=" * 70)
    print(f"  Month: {budget['month']}")
    print(f"  Budget: ${budget['budget']:.2f}")
    print(f"  Spent: ${budget['spent']:.2f}")
    print(f"  Remaining: ${budget['remaining']:.2f}")
    print(f"  Percent Used: {budget['percent_used']:.1f}%")
