"""
Alembic Environment Configuration for Patent Intelligence System

This module configures the Alembic migration environment, handling database
connections and migration execution for both online and offline modes.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import get_config

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate support
# Import your models here if using autogenerate
target_metadata = None


def get_database_url() -> str:
    """
    Get database URL from configuration or environment.

    Priority:
    1. Environment variable DATABASE_URL
    2. Individual environment variables (DB_HOST, DB_PORT, etc.)
    3. Configuration file
    4. Alembic config file

    Returns:
        Database connection URL.
    """
    # Check for DATABASE_URL environment variable
    if os.environ.get("DATABASE_URL"):
        return os.environ["DATABASE_URL"]

    # Try to build from individual environment variables
    db_host = os.environ.get("DB_HOST")
    db_port = os.environ.get("DB_PORT", "5432")
    db_name = os.environ.get("DB_NAME")
    db_user = os.environ.get("DB_USER")
    db_password = os.environ.get("DB_PASSWORD", "")

    if all([db_host, db_name, db_user]):
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Try to get from application config
    try:
        app_config = get_config()
        db_config = app_config.get_database_config()
        return (
            f"postgresql://{db_config['user']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
    except Exception:
        pass

    # Fall back to alembic.ini
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    This configures the context with just a URL and not an Engine,
    though an Engine is acceptable here as well. By skipping the Engine
    creation we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.
    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    In this scenario we need to create an Engine and associate a
    connection with the context.
    """
    url = get_database_url()

    connectable = create_engine(
        url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
