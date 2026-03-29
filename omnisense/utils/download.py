"""Download video from URL (YouTube, etc.) using yt-dlp."""

import subprocess
import tempfile
from pathlib import Path


def download_video(url: str, output_dir: str | None = None) -> str:
    out_dir = output_dir or tempfile.mkdtemp()
    out_path = str(Path(out_dir) / "video.mp4")
    subprocess.run(
        ["yt-dlp", "-f", "mp4", "-o", out_path, url],
        check=True,
    )
    return out_path
