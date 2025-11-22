"""
Logging configuration for the Patent Intelligence system.
Uses loguru for enhanced logging capabilities.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
    log_format: Optional[str] = None,
) -> None:
    """
    Configure the logging system.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to log file. If None, logs to stdout only.
        rotation: Log rotation policy (e.g., "10 MB", "1 day").
        retention: Log retention policy (e.g., "1 week", "30 days").
        log_format: Custom log format string.
    """
    # Remove default handler
    logger.remove()

    # Default format
    if log_format is None:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        )

    # Add stdout handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True,
    )

    # Add file handler if specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_file,
            format=log_format.replace("<green>", "")
            .replace("</green>", "")
            .replace("<level>", "")
            .replace("</level>", "")
            .replace("<cyan>", "")
            .replace("</cyan>", ""),
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )


def get_logger(name: str = "patent_intelligence"):
    """
    Get a logger instance with the specified name.

    Args:
        name: Logger name for identification.

    Returns:
        Logger instance.
    """
    return logger.bind(name=name)


# Initialize default logger
setup_logger()
