"""
omnisense/pipelines/audio.py

Audio pipeline: faster-whisper transcription, CPU-first design.
Targets HuggingFace Spaces free tier (no GPU assumed).

Key decisions:
  - faster-whisper (CTranslate2) instead of openai-whisper: 4x faster on CPU
  - compute_type="int8" on CPU: negligible accuracy loss, big speed gain
  - beam_size=1 (greedy decode): 2x faster than beam_size=5, ~same quality
  - vad_filter=True: skips silence → fewer segments, faster indexing
  - word_timestamps=False: segment-level is sufficient for search use case
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from faster_whisper import WhisperModel
from loguru import logger

# ── Data contract shared with search.py and app.py ────────────────────────────


@dataclass
class TranscriptChunk:
    """
    A single transcribed segment with its time bounds.
    This is the core data unit passed between all pipeline stages.
    """

    text: str
    start: float  # seconds from video start
    end: float  # seconds from video start
    chunk_id: int = 0


# ── Hardware detection ─────────────────────────────────────────────────────────


def _best_device() -> tuple[str, str]:
    """
    Detect available hardware and return (device, compute_type).

    Priority:
      CUDA GPU  → float16  (fastest, not expected on HF Spaces free tier)
      CPU       → int8     (best CPU performance via CTranslate2 quantization)

    Note: MPS (Apple Silicon) is intentionally not used here because
    faster-whisper's CTranslate2 backend does not support MPS.
    """
    if torch.cuda.is_available():
        logger.info("CUDA detected — using float16")
        return "cuda", "float16"
    logger.info("No GPU detected — using CPU with int8 quantization")
    return "cpu", "int8"


# ── Audio extraction ───────────────────────────────────────────────────────────


def extract_audio(video_path: str, output_dir: str = "/tmp") -> str:
    """
    Extract a 16kHz mono WAV audio track from any video file.

    Uses ffmpeg-python. The 16kHz / mono format is what Whisper expects;
    doing the conversion here (rather than letting Whisper do it internally)
    gives us a clean cached file we can inspect or reuse.

    Args:
        video_path:  Path to the uploaded video file.
        output_dir:  Directory to write the extracted WAV.

    Returns:
        Path to the extracted WAV file.
    """
    import ffmpeg

    audio_path = os.path.join(output_dir, "extracted_audio.wav")
    logger.info(f"Extracting audio from {video_path} → {audio_path}")

    (
        ffmpeg.input(video_path)
        .output(
            audio_path,
            ac=1,  # mono
            ar=16000,  # 16kHz — Whisper's native sample rate
            format="wav",
        )
        .overwrite_output()
        .run(quiet=True)
    )

    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info(f"Audio extracted: {size_mb:.1f} MB")
    return audio_path


# ── Transcription ──────────────────────────────────────────────────────────────

#: Estimated real-time factors on HF Spaces CPU free tier.
#: e.g. RTF=0.15 means 1 hour of audio takes ~9 minutes to transcribe.
MODEL_SPEED_GUIDE: dict[str, str] = {
    "tiny": "~2–4 min per hour of audio   · rough accuracy",
    "base": "~5–8 min per hour of audio   · good accuracy  ✓ recommended",
    "small": "~10–15 min per hour of audio · better accuracy",
    "medium": "~25–35 min per hour of audio · best CPU accuracy",
}


def transcribe(
    audio_path: str,
    model_size: str = "base",
) -> list[TranscriptChunk]:
    """
    Transcribe audio and return a list of segment-level TranscriptChunks.

    Each chunk carries the spoken text and its [start, end] time bounds in
    seconds. These are passed directly to the search pipeline for indexing.

    Args:
        audio_path:  Path to the extracted .wav file.
        model_size:  One of "tiny", "base", "small", "medium".
                     "base" is the recommended default for HF Spaces.

    Returns:
        List[TranscriptChunk] sorted by start time.

    Raises:
        FileNotFoundError: If audio_path does not exist.
        ValueError:        If model_size is not a valid Whisper model name.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    device, compute_type = _best_device()

    logger.info(
        f"Loading faster-whisper [{model_size}] | "
        f"device={device} | compute_type={compute_type}"
    )
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    logger.info("Transcribing…")
    segments_generator, info = model.transcribe(
        audio_path,
        word_timestamps=False,  # segment-level sufficient for search
        vad_filter=True,  # skip silence → cleaner segments
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        beam_size=1,  # greedy decode: 2x faster, negligible accuracy drop
    )

    logger.info(
        f"Language detected: {info.language} "
        f"(confidence {info.language_probability:.0%})"
    )

    chunks: list[TranscriptChunk] = []
    for i, seg in enumerate(segments_generator):
        text = seg.text.strip()
        if not text:
            continue
        chunk = TranscriptChunk(
            text=text,
            start=round(seg.start, 2),
            end=round(seg.end, 2),
            chunk_id=i,
        )
        chunks.append(chunk)
        logger.debug(f"  [{chunk.start:.1f}s → {chunk.end:.1f}s] {chunk.text[:80]}")

    if not chunks:
        logger.warning("Transcription produced zero segments — check your audio file.")

    logger.info(f"Transcription complete: {len(chunks)} segments")
    return chunks
