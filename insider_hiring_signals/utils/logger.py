"""
Logging configuration for Insider Activity + Hiring Signals System
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_size: str = "10 MB",
    retention: str = "1 week"
) -> logger:
    """
    Setup and configure loguru logger.

    Args:
        name: Logger name (for context)
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_size: Max size before rotation
        retention: How long to keep old logs

    Returns:
        Configured logger instance
    """
    # Remove default handler
    logger.remove()

    # Format for console output
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True
    )

    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        )

        logger.add(
            log_file,
            format=file_format,
            level=level,
            rotation=max_size,
            retention=retention,
            compression="zip"
        )

    if name:
        return logger.bind(name=name)

    return logger


# Create a default logger instance
default_logger = setup_logger(
    name="insider_signals",
    log_file="logs/insider_signals.log"
)
