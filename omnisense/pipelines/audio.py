"""
Audio transcription pipeline.

Uses OpenAI Whisper (via HuggingFace) to produce:
  - Full transcript text
  - Timestamped segments
  - NLP-ready text chunks
  - Detected language + confidence
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import whisper

from omnisense.config import CACHE_DIR, DEVICE, MAX_VIDEO_DURATION, MODELS
from omnisense.pipelines.base import BasePipeline
from omnisense.utils.logger import log
from omnisense.utils.media import (
    chunk_transcript,
    extract_audio_from_video,
    get_audio_duration,
)

# Video file extensions we can handle
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


class AudioPipeline(BasePipeline):
    """
    Transcribes audio/video files using OpenAI Whisper.

    Usage:
        pipeline = AudioPipeline(device="cpu")
        result = pipeline("/path/to/file.mp4")

    Result shape:
        {
            "transcript":  str,           # full joined text
            "segments":    list[dict],    # raw Whisper segments
            "chunks":      list[dict],    # NLP-ready chunks with timestamps
            "language":    str,           # detected language code e.g. "en"
            "duration":    float,         # audio duration in seconds
            "model":       str,           # model identifier used
        }
    """

    def __init__(self, device: str = DEVICE) -> None:
        super().__init__(device=device)
        self._model: whisper.Whisper | None = None
        self._model_name = (
            MODELS["whisper"].split("/")[-1].replace("whisper-", "")
        )  # "base"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download and cache Whisper weights."""
        log.info(f"Loading Whisper model '{self._model_name}'…")
        self._model = whisper.load_model(
            self._model_name,
            device=self._device_for_whisper(),
            download_root=str(CACHE_DIR / "whisper"),
        )
        log.info("Whisper model loaded ✓")

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(self, media_path: str | Path) -> dict[str, Any]:
        """
        Transcribe an audio or video file.

        Args:
            media_path: Path to audio (.wav, .mp3, …) or video (.mp4, …).

        Returns:
            Structured result dict (see class docstring).

        Raises:
            FileNotFoundError: File does not exist.
            ValueError: Duration exceeds MAX_VIDEO_DURATION.
            RuntimeError: Whisper transcription fails.
        """
        media_path = Path(media_path)
        self._validate_file(media_path)

        # If it's a video, extract audio track first
        audio_path = self._resolve_audio(media_path)

        # Guard against runaway files
        duration = get_audio_duration(audio_path)
        if duration > MAX_VIDEO_DURATION:
            raise ValueError(
                f"File duration {duration:.0f}s exceeds limit "
                f"of {MAX_VIDEO_DURATION}s. Set MAX_VIDEO_DURATION_SECONDS in .env."
            )

        log.info(f"Transcribing {audio_path.name} ({duration:.1f}s)…")
        raw = self._transcribe(audio_path)

        segments: list[dict] = raw.get("segments", [])
        transcript: str = raw.get("text", "").strip()
        language: str = raw.get("language", "unknown")
        chunks = chunk_transcript(segments)

        log.info(
            f"Transcription complete — {len(segments)} segments, "
            f"{len(chunks)} chunks, language={language}"
        )

        return {
            "transcript": transcript,
            "segments": segments,
            "chunks": chunks,
            "language": language,
            "duration": duration,
            "model": MODELS["whisper"],
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _validate_file(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")
        suffix = path.suffix.lower()
        if suffix not in VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {VIDEO_EXTENSIONS | AUDIO_EXTENSIONS}"
            )

    def _resolve_audio(self, path: Path) -> Path:
        """Return audio path — extract from video if needed."""
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return extract_audio_from_video(path, output_dir=CACHE_DIR / "audio")
        return path

    def _transcribe(self, audio_path: Path) -> dict:
        """Run Whisper inference with verbose logging disabled."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        try:
            return self._model.transcribe(
                str(audio_path),
                verbose=False,
                fp16=self.device == "cuda",
                word_timestamps=True,
            )
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

    def _device_for_whisper(self) -> str:
        """
        Whisper uses its own device strings.
        MPS (Apple Silicon) falls back to CPU — Whisper doesn't support MPS yet.
        """
        if self.device == "cuda" and torch.cuda.is_available():
            return "cuda"
        return "cpu"
