"""
Audio transcription pipeline.
Uses faster-whisper — CTranslate2 backend, no numba/llvmlite dependency,
2-4x faster than openai-whisper with identical accuracy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

from omnisense.config import CACHE_DIR, DEVICE, MAX_VIDEO_DURATION, MODELS
from omnisense.pipelines.base import BasePipeline
from omnisense.utils.logger import log
from omnisense.utils.media import (
    chunk_transcript,
    extract_audio_from_video,
    get_audio_duration,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


class AudioPipeline(BasePipeline):
    """
    Transcribes audio/video files using faster-whisper.

    Result shape:
        {
            "transcript":  str,
            "segments":    list[dict],
            "chunks":      list[dict],
            "language":    str,
            "duration":    float,
            "model":       str,
        }
    """

    def __init__(self, device: str = DEVICE) -> None:
        super().__init__(device=device)
        self._model: WhisperModel | None = None
        self._model_size = MODELS["whisper"].split("/")[-1].replace("whisper-", "")
        self._fw_device = "cuda" if device == "cuda" else "cpu"

    def load(self) -> None:
        """Download and cache faster-whisper weights."""
        log.info(f"Loading faster-whisper model '{self._model_size}'…")
        self._model = WhisperModel(
            self._model_size,
            device=self._fw_device,
            compute_type="int8",
            download_root=str(CACHE_DIR / "whisper"),
        )
        log.info("faster-whisper model loaded ✓")

    def run(self, media_path: str | Path) -> dict[str, Any]:
        """Transcribe an audio or video file."""
        media_path = Path(media_path)
        self._validate_file(media_path)

        audio_path = self._resolve_audio(media_path)

        duration = get_audio_duration(audio_path)
        if duration > MAX_VIDEO_DURATION:
            raise ValueError(
                f"Duration {duration:.0f}s exceeds limit of {MAX_VIDEO_DURATION}s."
            )

        log.info(f"Transcribing {audio_path.name} ({duration:.1f}s)…")
        segments_raw, info = self._model.transcribe(
            str(audio_path),
            beam_size=5,
            word_timestamps=True,
        )

        segments: list[dict] = []
        transcript_parts: list[str] = []
        for seg in segments_raw:
            segments.append(
                {
                    "text": seg.text.strip(),
                    "start": seg.start,
                    "end": seg.end,
                    "words": [
                        {"word": w.word, "start": w.start, "end": w.end}
                        for w in (seg.words or [])
                    ],
                }
            )
            transcript_parts.append(seg.text.strip())

        transcript = " ".join(transcript_parts)
        language = info.language
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
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return extract_audio_from_video(path, output_dir=CACHE_DIR / "audio")
        return path
