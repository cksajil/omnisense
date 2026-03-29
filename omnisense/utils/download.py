"""
omnisense/utils/download.py

Optional YouTube/URL video downloader using yt-dlp.
yt-dlp is NOT in requirements.txt — install separately if needed:
    pip install yt-dlp
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def download_video(url: str, output_dir: str | None = None) -> str:
    out_dir = output_dir or tempfile.mkdtemp()
    out_path = str(Path(out_dir) / "video.mp4")
    try:
        subprocess.run(
            ["yt-dlp", "-f", "mp4", "-o", out_path, url],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError("yt-dlp not found. Install it with: pip install yt-dlp")
    return out_path
