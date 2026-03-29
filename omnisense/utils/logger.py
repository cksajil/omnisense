"""
omnisense/utils/logger.py

Centralised logging with loguru.
Import `logger` from loguru directly in other modules,
or use this module to get a pre-configured instance.
"""

from __future__ import annotations

import sys

from loguru import logger

from omnisense.config import LOG_LEVEL

# Remove loguru's default stderr handler and replace with a formatted one
logger.remove()

logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    colorize=True,
)

logger.add(
    "logs/omnisense.log",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
    enqueue=True,  # thread-safe writes
)

# Alias kept for any legacy imports: `from omnisense.utils.logger import log`
log = logger
