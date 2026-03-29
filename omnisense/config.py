"""
omnisense/config.py

Minimal config for the temporal-search branch.
No pydantic_settings dependency — just plain constants + dotenv.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Whisper model size used when not overridden by the UI
WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "base")

# Log level for loguru — used by omnisense/utils/logger.py
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# HuggingFace token (optional — only needed for gated models)
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
