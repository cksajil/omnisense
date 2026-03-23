"""
Media extraction utilities.
Handles video → audio splitting and audio normalisation via ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from omnisense.utils.logger import log


def extract_audio_from_video(
    video_path: str | Path, output_dir: str | Path | None = None
) -> Path:
    """
    Extract audio track from a video file using ffmpeg.

    Args:
        video_path: Path to input video file.
        output_dir: Directory to write the .wav file.
                    Defaults to same directory as video.

    Returns:
        Path to the extracted .wav file.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If ffmpeg extraction fails.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir = Path(output_dir) if output_dir else video_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{video_path.stem}_audio.wav"

    if output_path.exists():
        log.debug(f"Audio already extracted, reusing: {output_path}")
        return output_path

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",  # no video
        "-acodec",
        "pcm_s16le",  # 16-bit PCM — Whisper's preferred format
        "-ar",
        "16000",  # 16kHz sample rate
        "-ac",
        "1",  # mono channel
        str(output_path),
    ]

    log.info(f"Extracting audio from {video_path.name}…")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    log.info(f"Audio extracted → {output_path}")
    return output_path


def get_audio_duration(audio_path: str | Path) -> float:
    """
    Return audio duration in seconds using ffprobe.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds as a float.
    """
    audio_path = Path(audio_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        log.warning(f"Could not determine duration for {audio_path}")
        return 0.0


def chunk_transcript(
    segments: list[dict],
    max_tokens: int = 512,
    overlap_tokens: int = 50,
) -> list[dict]:
    """
    Group Whisper segments into overlapping text chunks suitable for NLP models.

    Whisper returns fine-grained segments (often 3–10 words each).
    NLP models like BART need 200–500 word chunks. This function
    aggregates segments while preserving start/end timestamps.

    Args:
        segments: List of Whisper segment dicts with 'text', 'start', 'end'.
        max_tokens: Approximate max words per chunk (used as proxy for tokens).
        overlap_tokens: Words of overlap between consecutive chunks.

    Returns:
        List of chunk dicts: {'text', 'start', 'end', 'segment_ids'}
    """
    if not segments:
        return []

    chunks: list[dict] = []
    current_words: list[str] = []
    current_start: float = segments[0]["start"]
    current_seg_ids: list[int] = []

    for i, seg in enumerate(segments):
        words = seg["text"].strip().split()
        current_words.extend(words)
        current_seg_ids.append(i)

        if len(current_words) >= max_tokens:
            chunks.append(
                {
                    "text": " ".join(current_words),
                    "start": current_start,
                    "end": seg["end"],
                    "segment_ids": current_seg_ids.copy(),
                }
            )
            # Keep overlap words for context continuity
            current_words = current_words[-overlap_tokens:]
            current_start = seg["start"]
            current_seg_ids = [i]

    # Flush remaining words as the final chunk
    if current_words:
        chunks.append(
            {
                "text": " ".join(current_words),
                "start": current_start,
                "end": segments[-1]["end"],
                "segment_ids": current_seg_ids,
            }
        )

    log.debug(f"Chunked {len(segments)} segments → {len(chunks)} chunks")
    return chunks
