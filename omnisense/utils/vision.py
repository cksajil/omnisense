"""
Vision utilities — video frame extraction and image preprocessing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from omnisense.config import FRAME_SAMPLE_RATE
from omnisense.utils.logger import log


def extract_frames(
    video_path: str | Path,
    sample_rate: int = FRAME_SAMPLE_RATE,
    max_frames: int = 50,
) -> list[dict]:
    """
    Extract frames from a video file at a given sample rate.

    Args:
        video_path: Path to video file.
        sample_rate: Extract one frame every N seconds.
        max_frames: Maximum frames to extract (prevents memory issues).

    Returns:
        List of frame dicts: {'frame_id', 'timestamp', 'image': PIL.Image}
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_interval = int(fps * sample_rate)

    log.info(
        f"Extracting frames from {video_path.name} "
        f"({duration:.1f}s, {fps:.1f}fps, interval={frame_interval})"
    )

    frames: list[dict] = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            frames.append(
                {
                    "frame_id": len(frames),
                    "timestamp": round(frame_idx / fps, 2) if fps > 0 else 0,
                    "image": pil_image,
                    "width": pil_image.width,
                    "height": pil_image.height,
                }
            )

        frame_idx += 1

    cap.release()
    log.info(f"Extracted {len(frames)} frames from {video_path.name}")
    return frames


def resize_image(image: Image.Image, max_size: int = 800) -> Image.Image:
    """
    Resize image so longest side is at most max_size pixels.
    Preserves aspect ratio.
    """
    w, h = image.size
    if max(w, h) <= max_size:
        return image
    scale = max_size / max(w, h)
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array."""
    return np.array(image)


def save_frame(
    frame: dict,
    output_dir: str | Path,
) -> Path:
    """
    Save a frame dict's image to disk as JPEG.

    Args:
        frame: Frame dict from extract_frames().
        output_dir: Directory to save the image.

    Returns:
        Path to saved image.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"frame_{frame['frame_id']:04d}_{frame['timestamp']:.1f}s.jpg"
    frame["image"].save(str(path), "JPEG", quality=85)
    return path
