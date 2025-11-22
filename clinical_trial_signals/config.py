"""
Configuration settings for Clinical Trial Signal Detection System.
Copy config_template.py to config.py and fill in your actual values.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "clinical_trials")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class EmailConfig:
    """Email configuration for SendGrid."""
    sendgrid_api_key: str = os.getenv("SENDGRID_API_KEY", "")
    from_email: str = os.getenv("FROM_EMAIL", "alerts@clinicaltrials.local")
    to_emails: List[str] = field(default_factory=lambda: os.getenv("TO_EMAILS", "").split(",") if os.getenv("TO_EMAILS") else [])
    enabled: bool = os.getenv("EMAIL_ENABLED", "false").lower() == "true"


@dataclass
class ScraperConfig:
    """Configuration for data scrapers."""
    # ClinicalTrials.gov API
    clinicaltrials_base_url: str = "https://clinicaltrials.gov/api/v2"
    clinicaltrials_rate_limit: float = 0.5  # seconds between requests

    # SEC EDGAR API
    sec_base_url: str = "https://data.sec.gov"
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "ClinicalTrialSignals research@example.com")
    sec_rate_limit: float = 0.1  # SEC requires 10 requests/second max

    # General settings
    request_timeout: int = 30
    max_retries: int = 3


@dataclass
class SignalConfig:
    """Signal detection and scoring configuration."""
    # Signal weights (from spec) - Expanded to 20+ signal types
    weights: dict = field(default_factory=lambda: {
        # === POSITIVE SIGNALS ===
        # Trial Progress Signals
        "sites_added": 3,
        "early_enrollment": 3,
        "enrollment_increase": 2,
        "completion_date_accelerated": 3,
        "status_change_positive": 2,

        # Regulatory Signals
        "fda_breakthrough": 5,
        "fda_fast_track": 4,
        "fda_priority_review": 4,
        "fda_orphan_designation": 3,
        "patent_filed": 5,

        # Publication/Conference Signals
        "late_breaking_abstract": 5,
        "preprint_positive": 3,
        "ceo_presentation": 2,
        "conference_presentation": 2,

        # Financial Signals
        "insider_buying": 4,
        "sec_8k_positive": 3,
        "partnership_announced": 4,
        "funding_secured": 3,

        # === NEGATIVE SIGNALS ===
        # Trial Progress Signals
        "enrollment_extended": -3,
        "enrollment_decrease": -2,
        "completion_date_delayed": -3,
        "status_change_negative": -2,
        "sites_removed": -4,
        "endpoint_change": -5,

        # Regulatory Signals
        "clinical_hold": -5,
        "fda_complete_response": -4,
        "safety_signal": -4,

        # Publication/Conference Signals
        "preprint_negative": -3,
        "no_conference": -2,

        # Financial Signals
        "insider_selling": -4,
        "sec_8k_negative": -3,
        "risk_factor_increase": -3,

        # === NEUTRAL SIGNALS ===
        "status_change_neutral": 0,
        "data_update": 0,
    })

    # Scoring thresholds
    strong_buy_threshold: float = 7.0
    buy_threshold: float = 5.0
    short_threshold: float = 3.0
    confidence_threshold: float = 0.7

    # Historical accuracy (placeholder - would be updated with backtesting)
    historical_accuracy: float = 0.6


@dataclass
class MonitoringConfig:
    """Trial monitoring criteria from spec."""
    # Phase filters
    phases: List[str] = field(default_factory=lambda: ["PHASE3", "PHASE2/PHASE3"])

    # Market cap filters (in USD)
    min_market_cap: int = 500_000_000  # $500M
    max_market_cap: int = 5_000_000_000  # $5B

    # Number of trials to track
    max_trials: int = 20

    # Indications of interest (high unmet need)
    priority_indications: List[str] = field(default_factory=lambda: [
        "oncology", "cancer", "tumor", "carcinoma", "lymphoma", "leukemia",
        "rare disease", "orphan",
        "neurology", "alzheimer", "parkinson", "als", "multiple sclerosis",
        "immunology", "autoimmune",
    ])


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    email: EmailConfig = field(default_factory=EmailConfig)
    scraper: ScraperConfig = field(default_factory=ScraperConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Optional[str] = os.getenv("LOG_FILE", "clinical_trials.log")

    # Run mode
    dry_run: bool = os.getenv("DRY_RUN", "false").lower() == "true"


# Default configuration instance
config = Config()


def load_config_from_env() -> Config:
    """Load configuration from environment variables."""
    return Config()


def validate_config(cfg: Config) -> List[str]:
    """Validate configuration and return list of issues."""
    issues = []

    if not cfg.database.password and not cfg.dry_run:
        issues.append("Database password not set (DB_PASSWORD)")

    if cfg.email.enabled:
        if not cfg.email.sendgrid_api_key:
            issues.append("Email enabled but SENDGRID_API_KEY not set")
        if not cfg.email.to_emails or cfg.email.to_emails == [""]:
            issues.append("Email enabled but TO_EMAILS not set")

    if not cfg.scraper.sec_user_agent or "example.com" in cfg.scraper.sec_user_agent:
        issues.append("SEC_USER_AGENT should be set to a valid contact email per SEC requirements")

    return issues
