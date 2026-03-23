"""
Central configuration for OmniSense.
All settings are loaded from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env at import time so all modules see the values
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
ASSETS_DIR = ROOT_DIR / "assets"
CACHE_DIR = ROOT_DIR / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN: str | None = os.getenv("HF_TOKEN")

# ── Hardware ──────────────────────────────────────────────────────────────────
DEVICE: str = os.getenv("DEVICE", "cpu")  # "cuda" | "mps" | "cpu"

# ── Model identifiers (single source of truth) ────────────────────────────────
MODELS = {
    "whisper": "openai/whisper-base",
    "summarizer": "sshleifer/distilbart-cnn-6-6",
    "classifier": "facebook/bart-large-mnli",
    "ner": "dslim/bert-base-NER",
    "captioner": "Salesforce/blip-image-captioning-base",
    "clip": "openai/clip-vit-base-patch32",
    "embedder": "sentence-transformers/all-MiniLM-L6-v2",
    "detector": "facebook/detr-resnet-50",
}

# ── Pipeline settings ─────────────────────────────────────────────────────────
MAX_VIDEO_DURATION: int = int(os.getenv("MAX_VIDEO_DURATION_SECONDS", "600"))
FRAME_SAMPLE_RATE: int = 1  # extract 1 frame per second
CHUNK_SIZE_TOKENS: int = 512  # max tokens per NLP chunk
CHUNK_OVERLAP_TOKENS: int = 50  # overlap between chunks

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
