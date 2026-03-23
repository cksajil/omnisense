"""
Centralised logging with loguru.
Import `log` from here everywhere — never use print() in production code.
"""

import sys

from loguru import logger

from omnisense.config import LOG_LEVEL

logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level=LOG_LEVEL,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
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
)

log = logger
