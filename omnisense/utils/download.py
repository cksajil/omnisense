"""
omnisense/utils/download.py

Download a YouTube video (or any yt-dlp-supported URL) to a local .mp4 file.
Called by app.py when the user provides a URL instead of uploading a file.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

from loguru import logger


def is_youtube_url(url: str) -> bool:
    """Return True if the string looks like a YouTube or youtu.be URL."""
    url = url.strip()
    patterns = [
        r"^https?://(www\.)?youtube\.com/watch\?.*v=[\w-]+",
        r"^https?://youtu\.be/[\w-]+",
        r"^https?://(www\.)?youtube\.com/shorts/[\w-]+",
    ]
    return any(re.match(p, url) for p in patterns)


def download_video(
    url: str,
    output_dir: str | None = None,
    cookies_file: str | None = None,
) -> str:
    """
    Download a video from YouTube (or any yt-dlp-supported URL).

    Args:
        url:          A YouTube watch/shorts/youtu.be URL.
        output_dir:   Directory to save the downloaded file.
                      Defaults to a fresh temp directory.
        cookies_file: Path to a Netscape-format cookies.txt file.
                      Required on server deployments where YouTube
                      blocks anonymous requests with a bot-check error.
                      Export from your browser using an extension such as
                      "Get cookies.txt LOCALLY".

    Returns:
        Absolute path to the downloaded .mp4 file.

    Raises:
        RuntimeError: If yt-dlp is not installed or download fails.
        ValueError:   If the URL is not a recognised YouTube URL.
    """
    try:
        import yt_dlp  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "yt-dlp is not installed. " "Run: pip install yt-dlp>=2024.1.0"
        )

    if not is_youtube_url(url):
        raise ValueError(
            f"URL does not look like a YouTube link: {url!r}\n"
            "Supported formats: youtube.com/watch?v=..., youtu.be/..., youtube.com/shorts/..."
        )

    out_dir = output_dir or tempfile.mkdtemp()
    out_path = str(Path(out_dir) / "yt_video.mp4")

    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_path,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Android/iOS clients use a different API endpoint that does not
        # trigger YouTube's server-side bot-detection check. tv_embedded and
        # web are kept as fallbacks in case the mobile clients are unavailable.
        "extractor_args": {"youtube": {"player_client": ["android", "ios", "tv_embedded", "web"]}},
    }

    if cookies_file and Path(cookies_file).exists():
        ydl_opts["cookiefile"] = cookies_file
        logger.info(f"Using cookies file: {cookies_file}")

    logger.info(f"Downloading YouTube video: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", "unknown")
            duration = info.get("duration", 0)
            logger.info(f"Downloaded: '{title}' ({duration}s) → {out_path}")
    except Exception as e:
        raise RuntimeError(f"yt-dlp download failed: {e}") from e

    if not Path(out_path).exists():
        raise RuntimeError(
            f"Download seemed to succeed but file not found at {out_path}. "
            "yt-dlp may have saved it with a different name."
        )

    return out_path
