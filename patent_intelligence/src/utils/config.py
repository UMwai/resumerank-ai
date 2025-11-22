"""
Configuration management for the Patent Intelligence system.
Loads settings from YAML config file and environment variables.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


class Config:
    """Configuration manager that loads settings from YAML and environment variables."""

    _instance: Optional["Config"] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None) -> "Config":
        """Singleton pattern to ensure single config instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to YAML config file. If None, uses default location.
        """
        if self._initialized:
            return

        # Load environment variables from .env file
        self._load_env_files()

        # Determine config path
        if config_path is None:
            config_path = self._get_default_config_path()

        # Load YAML configuration
        self._load_yaml_config(config_path)

        self._initialized = True

    def _load_env_files(self) -> None:
        """Load environment variables from .env files."""
        # Try multiple locations for .env file
        possible_env_paths = [
            Path.cwd() / ".env",
            Path.cwd() / "config" / ".env",
            Path(__file__).parent.parent.parent / "config" / ".env",
            Path(__file__).parent.parent.parent / ".env",
        ]

        for env_path in possible_env_paths:
            if env_path.exists():
                load_dotenv(env_path)
                break

    def _get_default_config_path(self) -> str:
        """Get default configuration file path."""
        possible_paths = [
            Path.cwd() / "config" / "config.yaml",
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
            Path.cwd() / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise FileNotFoundError(
            "Configuration file not found. Please create config/config.yaml"
        )

    def _load_yaml_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file.
        """
        with open(config_path, "r") as f:
            raw_config = yaml.safe_load(f)

        # Replace environment variable placeholders
        self._config = self._resolve_env_vars(raw_config)

    def _resolve_env_vars(self, obj: Any) -> Any:
        """
        Recursively resolve environment variable placeholders in config.

        Args:
            obj: Configuration object (dict, list, or value).

        Returns:
            Configuration with environment variables resolved.
        """
        if isinstance(obj, dict):
            return {k: self._resolve_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._resolve_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            # Match ${VAR_NAME} pattern
            pattern = r"\$\{([^}]+)\}"
            matches = re.findall(pattern, obj)

            for var_name in matches:
                env_value = os.getenv(var_name, "")
                obj = obj.replace(f"${{{var_name}}}", env_value)

            return obj
        else:
            return obj

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.

        Args:
            key: Configuration key (e.g., "database.host").
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration as a dictionary."""
        return {
            "host": self.get("database.host", "localhost"),
            "port": self.get("database.port", 5432),
            "database": self.get("database.name", "patent_intelligence"),
            "user": self.get("database.user", "patent_user"),
            "password": self.get("database.password", ""),
        }

    def get_database_url(self) -> str:
        """Get SQLAlchemy database URL."""
        db = self.get_database_config()
        return (
            f"postgresql://{db['user']}:{db['password']}"
            f"@{db['host']}:{db['port']}/{db['database']}"
        )

    def get_email_config(self) -> Dict[str, Any]:
        """Get email configuration as a dictionary."""
        return {
            "enabled": self.get("email.enabled", False),
            "smtp_server": self.get("email.smtp_server", "smtp.gmail.com"),
            "smtp_port": self.get("email.smtp_port", 587),
            "use_tls": self.get("email.use_tls", True),
            "sender_email": self.get("email.sender_email", ""),
            "sender_password": self.get("email.sender_password", ""),
            "recipients": self.get("email.recipients", []),
        }

    def get_scoring_weights(self) -> Dict[str, float]:
        """Get scoring model weights."""
        return {
            "patent_expired": self.get("scoring.patent_expired_weight", 0.40),
            "no_litigation": self.get("scoring.no_litigation_weight", 0.30),
            "anda_approved": self.get("scoring.anda_approved_weight", 0.20),
            "no_extension": self.get("scoring.no_extension_weight", 0.10),
        }

    def get_calendar_settings(self) -> Dict[str, Any]:
        """Get patent cliff calendar settings."""
        return {
            "forward_months": self.get("calendar.forward_months", 12),
            "high_certainty_threshold": self.get(
                "calendar.high_certainty_threshold", 80
            ),
            "medium_certainty_threshold": self.get(
                "calendar.medium_certainty_threshold", 60
            ),
            "blockbuster_threshold": self.get(
                "calendar.blockbuster_threshold", 1_000_000_000
            ),
            "high_value_threshold": self.get("calendar.high_value_threshold", 500_000_000),
            "medium_value_threshold": self.get(
                "calendar.medium_value_threshold", 100_000_000
            ),
        }

    @property
    def top_drugs_count(self) -> int:
        """Get number of top drugs to track."""
        return self.get("etl.top_drugs_count", 50)

    @property
    def batch_size(self) -> int:
        """Get ETL batch size."""
        return self.get("etl.batch_size", 100)

    @property
    def max_retries(self) -> int:
        """Get maximum retry attempts."""
        return self.get("etl.max_retries", 3)

    def __repr__(self) -> str:
        return f"Config(keys={list(self._config.keys())})"


# Convenience function to get config instance
def get_config(config_path: Optional[str] = None) -> Config:
    """
    Get the configuration singleton instance.

    Args:
        config_path: Optional path to config file.

    Returns:
        Config instance.
    """
    return Config(config_path)
