"""
Configuration loader for Insider Activity + Hiring Signals System
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class Config:
    """Configuration manager that loads from YAML file and environment variables."""

    _instance: Optional['Config'] = None
    _config: Dict[str, Any] = {}

    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path
            config_path = os.environ.get(
                'INSIDER_SIGNALS_CONFIG',
                Path(__file__).parent.parent / 'config' / 'config.yaml'
            )

        config_path = Path(config_path)

        if config_path.exists():
            with open(config_path, 'r') as f:
                self._config = yaml.safe_load(f) or {}
        else:
            # Try template file
            template_path = config_path.with_suffix('.yaml.template')
            if template_path.exists():
                with open(template_path, 'r') as f:
                    self._config = yaml.safe_load(f) or {}
                print(f"Warning: Using template config. Copy {template_path} to {config_path}")
            else:
                self._config = {}
                print(f"Warning: No config file found at {config_path}")

        # Override with environment variables
        self._apply_env_overrides()

    def _apply_env_overrides(self):
        """Apply environment variable overrides."""
        env_mappings = {
            'DATABASE_HOST': ('database', 'host'),
            'DATABASE_PORT': ('database', 'port'),
            'DATABASE_NAME': ('database', 'name'),
            'DATABASE_USER': ('database', 'user'),
            'DATABASE_PASSWORD': ('database', 'password'),
            'SEC_USER_AGENT': ('sec_edgar', 'user_agent'),
            'ANTHROPIC_API_KEY': ('ai', 'anthropic_api_key'),
            'OPENAI_API_KEY': ('ai', 'openai_api_key'),
            'SMTP_SERVER': ('email', 'smtp_server'),
            'SMTP_PORT': ('email', 'smtp_port'),
            'SENDER_EMAIL': ('email', 'sender_email'),
            'SENDER_PASSWORD': ('email', 'sender_password'),
        }

        for env_var, path in env_mappings.items():
            value = os.environ.get(env_var)
            if value:
                self._set_nested(path, value)

    def _set_nested(self, path: tuple, value: Any):
        """Set a nested config value."""
        d = self._config
        for key in path[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[path[-1]] = value

    def _get_nested(self, path: tuple, default: Any = None) -> Any:
        """Get a nested config value."""
        d = self._config
        for key in path:
            if isinstance(d, dict) and key in d:
                d = d[key]
            else:
                return default
        return d

    @property
    def database(self) -> Dict[str, Any]:
        """Database configuration."""
        return self._config.get('database', {})

    @property
    def database_url(self) -> str:
        """Construct database URL."""
        db = self.database
        return (
            f"postgresql://{db.get('user', 'postgres')}:"
            f"{db.get('password', '')}@"
            f"{db.get('host', 'localhost')}:"
            f"{db.get('port', 5432)}/"
            f"{db.get('name', 'insider_signals')}"
        )

    @property
    def sec_edgar(self) -> Dict[str, Any]:
        """SEC EDGAR configuration."""
        return self._config.get('sec_edgar', {})

    @property
    def sec_user_agent(self) -> str:
        """SEC EDGAR user agent (required by SEC)."""
        return self.sec_edgar.get('user_agent', 'Anonymous anonymous@example.com')

    @property
    def ai(self) -> Dict[str, Any]:
        """AI configuration."""
        return self._config.get('ai', {})

    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Anthropic API key."""
        return self.ai.get('anthropic_api_key')

    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key."""
        return self.ai.get('openai_api_key')

    @property
    def email(self) -> Dict[str, Any]:
        """Email configuration."""
        return self._config.get('email', {})

    @property
    def scraping(self) -> Dict[str, Any]:
        """Scraping configuration."""
        return self._config.get('scraping', {})

    @property
    def signals(self) -> Dict[str, Any]:
        """Signal configuration."""
        return self._config.get('signals', {})

    @property
    def watchlist(self) -> List[str]:
        """Biotech watchlist tickers."""
        return self._config.get('watchlist', [])

    @property
    def institutional_investors(self) -> List[Dict[str, str]]:
        """Institutional investors to track."""
        return self._config.get('institutional_investors', [])

    @property
    def logging(self) -> Dict[str, Any]:
        """Logging configuration."""
        return self._config.get('logging', {})

    def get(self, key: str, default: Any = None) -> Any:
        """Get a top-level config value."""
        return self._config.get(key, default)

    def get_nested(self, *path, default: Any = None) -> Any:
        """Get a nested config value by path."""
        return self._get_nested(path, default)

    @classmethod
    def reload(cls, config_path: Optional[str] = None):
        """Reload configuration."""
        cls._instance = None
        return cls(config_path)


# Convenience function
def get_config(config_path: Optional[str] = None) -> Config:
    """Get or create the singleton config instance."""
    return Config(config_path)
