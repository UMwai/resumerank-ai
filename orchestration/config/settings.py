"""
Orchestration Configuration Settings

Central configuration for all pipeline orchestration settings.
Supports dev/staging/production environments.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from pathlib import Path


class Environment(Enum):
    """Deployment environment."""
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ScheduleConfig:
    """Schedule configuration for pipelines."""
    # Clinical Trial Signals - Daily at 6:00 PM ET (after ClinicalTrials.gov updates)
    clinical_trial_cron: str = "0 18 * * *"  # 6:00 PM ET
    clinical_trial_timezone: str = "America/New_York"

    # Patent/IP Intelligence - Weekly on Monday at 8:00 AM ET
    patent_ip_cron: str = "0 8 * * 1"  # Monday 8:00 AM ET
    patent_ip_timezone: str = "America/New_York"

    # Form 4 Insider Trading - Every 30 minutes during market hours (9:30 AM - 4:00 PM ET)
    form4_cron: str = "*/30 9-16 * * 1-5"  # Every 30 min, Mon-Fri, 9 AM - 4 PM
    form4_timezone: str = "America/New_York"

    # 13F Institutional Holdings - Quarterly (45 days after quarter end)
    # Feb 14, May 15, Aug 14, Nov 14
    form13f_cron: str = "0 8 14 2,5,8,11 *"  # 8 AM on the 14th of Feb, May, Aug, Nov
    form13f_timezone: str = "America/New_York"

    # Job Postings - Daily at 9:00 AM ET
    job_postings_cron: str = "0 9 * * *"  # 9:00 AM ET
    job_postings_timezone: str = "America/New_York"


@dataclass
class RetryConfig:
    """Retry configuration for failed tasks."""
    max_retries: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    retry_exponential_backoff: bool = True
    max_retry_delay_seconds: int = 3600  # 1 hour


@dataclass
class AlertConfig:
    """Alert configuration."""
    # Email settings
    email_enabled: bool = True
    email_smtp_server: str = ""
    email_smtp_port: int = 587
    email_smtp_user: str = ""
    email_smtp_password: str = ""
    email_from: str = ""
    email_recipients: List[str] = field(default_factory=list)
    email_high_confidence_threshold: float = 7.0

    # Slack settings
    slack_enabled: bool = True
    slack_webhook_url: str = ""
    slack_channel: str = "#investment-signals"
    slack_mention_users: List[str] = field(default_factory=list)

    # SMS settings (Twilio)
    sms_enabled: bool = False
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sms_recipients: List[str] = field(default_factory=list)
    sms_critical_only: bool = True

    # Deduplication settings
    alert_dedup_window_hours: int = 24
    alert_dedup_key_fields: List[str] = field(default_factory=lambda: ["signal_type", "ticker", "signal_id"])


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "investment_signals"
    username: str = ""
    password: str = ""
    pool_size: int = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RateLimitConfig:
    """API rate limiting configuration."""
    sec_edgar_requests_per_second: float = 10.0
    clinicaltrials_requests_per_second: float = 3.0
    uspto_requests_per_second: float = 5.0
    linkedin_requests_per_hour: int = 100
    indeed_requests_per_hour: int = 50


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    health_check_interval_seconds: int = 300
    metrics_enabled: bool = True
    metrics_port: int = 9090
    data_freshness_threshold_hours: int = 24

    # Cost tracking
    cost_tracking_enabled: bool = True
    cost_alert_threshold_daily: float = 50.0
    cost_alert_threshold_monthly: float = 1000.0


@dataclass
class AirflowConfig:
    """Airflow-specific configuration."""
    dags_folder: str = "/opt/airflow/dags"
    executor: str = "LocalExecutor"  # LocalExecutor, CeleryExecutor, KubernetesExecutor
    parallelism: int = 16
    dag_concurrency: int = 4
    max_active_runs_per_dag: int = 1
    load_examples: bool = False

    # Webserver
    webserver_host: str = "0.0.0.0"
    webserver_port: int = 8080

    # Scheduler
    scheduler_heartbeat_sec: int = 5
    min_file_process_interval: int = 30


@dataclass
class CloudConfig:
    """Cloud deployment configuration."""
    provider: str = "aws"  # aws, gcp, azure
    region: str = "us-east-1"

    # AWS specific
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""
    aws_ecs_cluster: str = ""

    # GCP specific
    gcp_project_id: str = ""
    gcp_service_account_key: str = ""
    gcp_storage_bucket: str = ""
    gcp_cloud_run_region: str = "us-central1"

    # Secrets Manager
    use_secrets_manager: bool = True
    secrets_prefix: str = "investment-signals"


@dataclass
class OrchestrationConfig:
    """Main orchestration configuration."""
    environment: Environment = Environment.DEV
    project_root: str = "/Users/waiyang/Desktop/repo/dreamers-v2"

    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    rate_limits: RateLimitConfig = field(default_factory=RateLimitConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    airflow: AirflowConfig = field(default_factory=AirflowConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)

    @classmethod
    def from_environment(cls, env: str = None) -> "OrchestrationConfig":
        """
        Load configuration from environment variables.

        Args:
            env: Environment name (dev/staging/production)

        Returns:
            OrchestrationConfig instance
        """
        env_name = env or os.getenv("ORCHESTRATION_ENV", "dev")
        environment = Environment(env_name.lower())

        config = cls(environment=environment)

        # Load from environment variables
        config._load_from_env()

        return config

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Database
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", str(self.database.port)))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)

        # Email
        self.alerts.email_smtp_server = os.getenv("SMTP_SERVER", self.alerts.email_smtp_server)
        self.alerts.email_smtp_port = int(os.getenv("SMTP_PORT", str(self.alerts.email_smtp_port)))
        self.alerts.email_smtp_user = os.getenv("SMTP_USER", self.alerts.email_smtp_user)
        self.alerts.email_smtp_password = os.getenv("SMTP_PASSWORD", self.alerts.email_smtp_password)
        self.alerts.email_from = os.getenv("EMAIL_FROM", self.alerts.email_from)
        recipients = os.getenv("EMAIL_RECIPIENTS", "")
        if recipients:
            self.alerts.email_recipients = [r.strip() for r in recipients.split(",")]

        # Slack
        self.alerts.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", self.alerts.slack_webhook_url)
        self.alerts.slack_channel = os.getenv("SLACK_CHANNEL", self.alerts.slack_channel)

        # Twilio
        self.alerts.twilio_account_sid = os.getenv("TWILIO_ACCOUNT_SID", self.alerts.twilio_account_sid)
        self.alerts.twilio_auth_token = os.getenv("TWILIO_AUTH_TOKEN", self.alerts.twilio_auth_token)
        self.alerts.twilio_from_number = os.getenv("TWILIO_FROM_NUMBER", self.alerts.twilio_from_number)
        sms_recipients = os.getenv("SMS_RECIPIENTS", "")
        if sms_recipients:
            self.alerts.sms_recipients = [r.strip() for r in sms_recipients.split(",")]

        # AWS
        self.cloud.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", self.cloud.aws_access_key_id)
        self.cloud.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", self.cloud.aws_secret_access_key)
        self.cloud.aws_s3_bucket = os.getenv("AWS_S3_BUCKET", self.cloud.aws_s3_bucket)
        self.cloud.region = os.getenv("AWS_REGION", self.cloud.region)

        # GCP
        self.cloud.gcp_project_id = os.getenv("GCP_PROJECT_ID", self.cloud.gcp_project_id)
        self.cloud.gcp_service_account_key = os.getenv("GCP_SERVICE_ACCOUNT_KEY", self.cloud.gcp_service_account_key)
        self.cloud.gcp_storage_bucket = os.getenv("GCP_STORAGE_BUCKET", self.cloud.gcp_storage_bucket)

        # Project root
        self.project_root = os.getenv("PROJECT_ROOT", self.project_root)


def get_config(env: str = None) -> OrchestrationConfig:
    """
    Get the orchestration configuration.

    Args:
        env: Environment name (dev/staging/production)

    Returns:
        OrchestrationConfig instance
    """
    return OrchestrationConfig.from_environment(env)


# Environment-specific configuration overrides
ENV_CONFIGS: Dict[str, Dict] = {
    "dev": {
        "alerts": {
            "email_enabled": False,
            "slack_enabled": False,
            "sms_enabled": False,
        },
        "monitoring": {
            "cost_tracking_enabled": False,
        },
        "database": {
            "host": "localhost",
            "database": "investment_signals_dev",
        },
    },
    "staging": {
        "alerts": {
            "email_enabled": True,
            "slack_enabled": True,
            "sms_enabled": False,
        },
        "monitoring": {
            "cost_tracking_enabled": True,
            "cost_alert_threshold_daily": 10.0,
        },
        "database": {
            "host": "localhost",
            "database": "investment_signals_staging",
        },
    },
    "production": {
        "alerts": {
            "email_enabled": True,
            "slack_enabled": True,
            "sms_enabled": True,
        },
        "monitoring": {
            "cost_tracking_enabled": True,
            "cost_alert_threshold_daily": 50.0,
            "cost_alert_threshold_monthly": 1000.0,
        },
        "airflow": {
            "executor": "CeleryExecutor",
            "parallelism": 32,
        },
    },
}
