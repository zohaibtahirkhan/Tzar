"""
Structured logging setup using Loguru.
Logs to stderr (console) and rotating files.
Sensitive data (raw audio) is never logged.
"""
import sys
from pathlib import Path

from loguru import logger

from app.config import settings


def setup_logging(level: str = "INFO") -> None:
    logger.remove()  # Remove default handler

    # Console — coloured
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{module}</cyan>:<cyan>{function}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # File — rotating, no raw audio
    log_file = settings.log_dir / "assistant_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        level="DEBUG",
        rotation="00:00",      # New file each day
        retention="7 days",
        compression="gz",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {module}:{function}:{line} — {message}",
    )

    logger.info("Logging initialized. Level: {}", level)
