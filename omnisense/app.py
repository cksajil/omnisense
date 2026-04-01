"""
omnisense/app.py

OmniSense - Temporal Video Search
Designed to run on HuggingFace Spaces CPU free tier (no GPU required).

Input modes:
  1. Paste a YouTube URL  → yt-dlp downloads to a temp .mp4
  2. Upload a video file  → used directly

Updated playback behavior:
  - When a user selects a search hit, the app extracts that exact video segment
    into a short preview clip.
  - The preview clip is embedded directly in the UI and autoplayed once.
  - This avoids relying on browser seek behavior or Gradio time-based seeking.
"""

from __future__ import annotations

# torch MUST be the first heavy import to register OpenMP before ctranslate2.
# Prevents segfault on macOS. Do not move this import.
import torch  # noqa: F401

import base64
import os
import shutil
import subprocess
import tempfile
import uuid

import gradio as gr
from loguru import logger

from omnisense.pipelines.audio import (
    MODEL_SPEED_GUIDE,
    TranscriptChunk,
    extract_audio,
    transcribe,
)
from omnisense.pipelines.search import SearchHit, TranscriptSearchIndex
from omnisense.utils.download import download_video, is_youtube_url


# ── Module-level session state ─────────────────────────────────────────────────

_index: TranscriptSearchIndex | None = None
_chunks: list[TranscriptChunk] = []
_video_path: str | None = None
_preview_dir: str = tempfile.mkdtemp(prefix="omnisense_preview_")
_last_clip_path: str | None = None


# ── Event handlers ─────────────────────────────────────────────────────────────


def handle_process(
    video_file: str | None,
    youtube_url: str,
    cookies_file: str | None,
    model_size: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, gr.update]:
    global _index, _chunks, _video_path, _last_clip_path

    youtube_url = (youtube_url or "").strip()
    _last_clip_path = None

    # Resolve the video source — URL takes priority over upload
    if youtube_url:
        if not is_youtube_url(youtube_url):
            return (
                "That doesn't look like a valid YouTube URL. "
                "Expected format: https://www.youtube.com/watch?v=...",
                gr.update(interactive=False),
            )
        try:
            progress(0.05, desc="Downloading YouTube video...")
            source_path = download_video(youtube_url, cookies_file=cookies_file)
        except RuntimeError as e:
            return f"Download failed: {e}", gr.update(interactive=False)
    elif video_file is not None:
        source_path = video_file
    else:
        return (
            "Please paste a YouTube URL or upload a video file first.",
            gr.update(interactive=False),
        )

    _video_path = source_path

    try:
        progress(0.20, desc="Extracting audio track...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = extract_audio(source_path, output_dir=tmpdir)
            progress(
                0.40,
                desc=f"Transcribing with Whisper [{model_size}] — please wait...",
            )
            _chunks = transcribe(audio_path, model_size=model_size)

        if not _chunks:
            return (
                "Transcription returned no segments. "
                "Check that the video has audible speech.",
                gr.update(interactive=False),
            )

        progress(0.85, desc="Building semantic search index...")
        _index = TranscriptSearchIndex()
        _index.build(_chunks)

        duration_s = _chunks[-1].end if _chunks else 0
        m, s = divmod(int(duration_s), 60)
        status = (
            f"Ready! Indexed **{len(_chunks)} segments** "
            f"across **{m}m {s}s** of audio.\n\n"
            f"Type anything in the search box below to find it."
        )
        progress(1.0)
        return status, gr.update(interactive=True)

    except Exception as e:
        logger.exception("Error during processing")
        return f"Error: {e}", gr.update(interactive=False)


def handle_search(
    query: str,
    top_k: int,
    min_score: float,
) -> tuple[str, gr.update]:
    if _index is None or not _index.is_ready:
        return (
            "<p style='color:orange;padding:12px'>Process a video first.</p>",
            gr.update(visible=False),
        )

    query = query.strip()
    if not query:
        return (
            "<p style='color:#888;padding:12px'>Enter a search query above.</p>",
            gr.update(visible=False),
        )

    hits = _index.search(query, top_k=int(top_k), min_score=float(min_score))

    if not hits:
        html = (
            "<div style='padding:16px;border-radius:8px;"
            "background:#fff8e1;border:1px solid #ffe082'>"
            f'<strong>No matches found</strong> for <em>"{query}"</em><br>'
            "<small style='color:#666;margin-top:6px;display:block'>"
            "Try rephrasing, or lower the Min Similarity slider."
            "</small></div>"
        )
        return html, gr.update(visible=False)

    cards_html = _build_results_html(hits, query)
    radio_choices = [_hit_to_label(h) for h in hits]

    return (
        cards_html,
        gr.update(choices=radio_choices, value=None, visible=True),
    )


def handle_hit_selected(label: str) -> tuple[gr.update, gr.update]:
    """
    User selects a search hit.
    Extract the selected segment into a short clip and autoplay it once.
    """
    if not label or _video_path is None:
        return gr.update(value="", visible=False), gr.update(value="")

    start_sec, end_sec = _parse_times_from_label(label)
    start_fmt = _fmt_time(start_sec)
    end_fmt = _fmt_time(end_sec)

    logger.info(f"Selected hit: clip {start_sec}s -> {end_sec}s")

    try:
        clip_path = _create_preview_clip(_video_path, start_sec, end_sec)
        clip_data_url = _video_file_to_data_url(clip_path)

        playback_html = f"""
        <div style="
            background:#ffffff;
            border:1px solid #dbeafe;
            border-radius:12px;
            padding:16px;
            box-shadow:0 1px 4px rgba(0,0,0,0.06);
            font-family:system-ui,-apple-system,sans-serif;
        ">
            <div style="
                background:#1d4ed8;
                color:#fff;
                border-radius:10px;
                padding:14px 18px;
                margin-bottom:14px;
                text-align:center;
            ">
                <div style="font-size:13px;opacity:0.9;margin-bottom:4px;">
                    Playing selected segment
                </div>
                <div style="font-size:34px;font-weight:800;letter-spacing:1px;">
                    {start_fmt} → {end_fmt}
                </div>
                <div style="font-size:13px;opacity:0.9;margin-top:4px;">
                    This preview clip starts automatically and plays once.
                </div>
            </div>

            <video
                controls
                autoplay
                playsinline
                style="width:100%;border-radius:10px;background:#000;"
            >
                <source src="{clip_data_url}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """

        return (
            gr.update(value=playback_html, visible=True),
            gr.update(value=""),
        )

    except Exception as e:
        logger.exception("Error while creating preview clip")
        error_html = f"""
        <div style="
            padding:14px 16px;
            border-radius:10px;
            background:#fef2f2;
            border:1px solid #fecaca;
            color:#991b1b;
            font-family:system-ui,-apple-system,sans-serif;
        ">
            <strong>Could not generate preview clip.</strong><br>
            <span style="font-size:13px;">{e}</span>
        </div>
        """
        return (
            gr.update(value=error_html, visible=True),
            gr.update(value=""),
        )


def handle_model_change(model_size: str) -> str:
    return f"**{model_size}** -- {MODEL_SPEED_GUIDE.get(model_size, '')}"


def handle_clear() -> tuple:
    global _index, _chunks, _video_path, _last_clip_path, _preview_dir

    _index = None
    _chunks = []
    _video_path = None
    _last_clip_path = None

    try:
        if os.path.isdir(_preview_dir):
            shutil.rmtree(_preview_dir, ignore_errors=True)
    except Exception:
        logger.exception("Failed to remove preview directory during clear")

    _preview_dir = tempfile.mkdtemp(prefix="omnisense_preview_")

    return (
        None,  # video_input
        "",  # youtube_url
        None,  # cookies_upload
        "base",  # model_choice
        f"**base** -- {MODEL_SPEED_GUIDE['base']}",  # model_hint
        "Upload a video and click Transcribe and Index to begin.",  # status_md
        gr.update(interactive=False),  # search_btn
        "",  # query_box
        "",  # results_html
        gr.update(choices=[], visible=False),  # hit_selector
        gr.update(value=""),  # seek_banner
        gr.update(value="", visible=False),  # playback_html
    )


# ── Clip helpers ───────────────────────────────────────────────────────────────


def _create_preview_clip(source_path: str, start_sec: float, end_sec: float) -> str:
    global _last_clip_path, _preview_dir

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video file not found: {source_path}")

    # Guard against invalid or tiny segments
    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 0.2, float(end_sec))
    duration = end_sec - start_sec

    clip_name = f"clip_{uuid.uuid4().hex}.mp4"
    clip_path = os.path.join(_preview_dir, clip_name)

    # Re-encode for reliable browser playback and exact segment extraction
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        source_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        clip_path,
    ]

    logger.info(f"Creating preview clip: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0 or not os.path.isfile(clip_path):
        raise RuntimeError(
            "ffmpeg failed to extract the selected segment.\n"
            f"{result.stderr[-1200:]}"
        )

    _last_clip_path = clip_path
    return clip_path


def _video_file_to_data_url(video_path: str) -> str:
    with open(video_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:video/mp4;base64,{encoded}"


# ── Label / time helpers ───────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _hit_to_label(hit: SearchHit) -> str:
    preview = hit.chunk.text[:100]
    if len(hit.chunk.text) > 100:
        preview += "..."
    return (
        f"#{hit.rank}  "
        f"[{_fmt_time(hit.chunk.start)} -> {_fmt_time(hit.chunk.end)}]  "
        f"score:{hit.score:.0%}  --  {preview}"
    )


def _parse_times_from_label(label: str) -> tuple[float, float]:
    range_text = label.split("[")[1].split("]")[0]
    start_str, end_str = [x.strip() for x in range_text.split("->")]

    def _to_seconds(ts: str) -> int:
        parts = ts.split(":")
        return int(parts[0]) * 60 + int(parts[1])

    return float(_to_seconds(start_str)), float(_to_seconds(end_str))


# ── Result card HTML ───────────────────────────────────────────────────────────


def _build_results_html(hits: list[SearchHit], query: str) -> str:

    def _score_color(score: float) -> str:
        if score >= 0.60:
            return "#16a34a"
        if score >= 0.40:
            return "#ca8a04"
        return "#dc2626"

    header = (
        "<div style='font-family:system-ui,-apple-system,sans-serif;padding:2px'>"
        "<p style='color:#475569;font-size:14px;margin:0 0 14px 0'>"
        f"<strong>{len(hits)} result{'s' if len(hits) != 1 else ''}</strong> for "
        f"&ldquo;<em style='color:#2563eb'>{query}</em>&rdquo;"
        "&nbsp;&nbsp;·&nbsp;&nbsp;"
        "<span style='color:#94a3b8'>select a result below to play that exact segment</span>"
        "</p>"
    )

    cards = ""
    for h in hits:
        color = _score_color(h.score)
        bar_pct = int(h.score * 100)
        cards += f"""
        <div style="
            border:1px solid #e2e8f0;border-radius:10px;
            padding:14px 18px;margin-bottom:12px;
            background:#ffffff;box-shadow:0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:8px;">
                <span style="font-weight:700;font-size:15px;color:#1e293b">
                    #{h.rank}
                    <span style="background:#1d4ed8;color:#fff;border-radius:5px;
                                 padding:2px 10px;font-size:13px;margin-left:8px;
                                 font-weight:600;">
                        {_fmt_time(h.chunk.start)} - {_fmt_time(h.chunk.end)}
                    </span>
                </span>
                <span style="font-size:13px;color:#64748b">
                    match &nbsp;
                    <strong style="color:{color};font-size:15px">{h.score:.0%}</strong>
                </span>
            </div>
            <div style="background:#f1f5f9;border-radius:4px;height:5px;margin-bottom:10px">
                <div style="width:{bar_pct}%;background:{color};height:5px;
                            border-radius:4px;"></div>
            </div>
            <p style="margin:0;color:#334155;font-size:14px;line-height:1.65">
                {h.chunk.text}
            </p>
        </div>
        """

    return header + cards + "</div>"


# ── Gradio UI ──────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="OmniSense - Temporal Video Search",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
            .gradio-container { max-width: 1080px !important; }
            #model-radio label span { font-size: 13px !important; }
            footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            """
        # OmniSense - Temporal Video Search
        **Find exactly when something was said in a video. No more scrubbing.**

        Paste a YouTube URL or upload a video, transcribe it, search in plain English,
        then click a result to instantly preview that exact segment.

        > Runs fully on CPU. No GPU required.
        > Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + [FAISS](https://github.com/facebookresearch/faiss)
        """
        )

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Step 1 - Load a video")

                gr.Markdown("**Option A — paste a YouTube URL**")
                youtube_url = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                )
                gr.Markdown(
                    "<small style='color:#94a3b8'>"
                    "Paste a URL and click Transcribe — no file upload needed.<br>"
                    "On hosted deployments YouTube may require cookies to bypass "
                    "bot-detection. Export <code>cookies.txt</code> from your browser "
                    "using the <em>Get cookies.txt LOCALLY</em> extension and upload "
                    "it below."
                    "</small>"
                )
                cookies_upload = gr.File(
                    label="YouTube cookies.txt (optional — only needed on servers)",
                    file_types=[".txt"],
                    type="filepath",
                )

                gr.Markdown("**Option B — upload a video file**")
                video_input = gr.Video(label="Upload video", height=200)

                model_choice = gr.Radio(
                    choices=list(MODEL_SPEED_GUIDE.keys()),
                    value="base",
                    label="Whisper model (speed vs accuracy)",
                    elem_id="model-radio",
                )
                model_hint = gr.Markdown(f"**base** -- {MODEL_SPEED_GUIDE['base']}")

                with gr.Row():
                    process_btn = gr.Button(
                        "Transcribe and Index", variant="primary", size="lg"
                    )
                    clear_btn = gr.Button("Clear", variant="secondary", size="lg")

                status_md = gr.Markdown(
                    "Paste a YouTube URL or upload a video, then click Transcribe and Index."
                )

            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Step 2 - Search")

                query_box = gr.Textbox(
                    label="What are you looking for?",
                    placeholder=(
                        'e.g.  "climate change"  or  '
                        '"when did he mention the budget?"'
                    ),
                    lines=3,
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Max results",
                    )
                    min_score_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.90,
                        value=0.30,
                        step=0.05,
                        label="Min similarity",
                    )

                gr.Markdown(
                    "<small style='color:#94a3b8'>"
                    "Lower Min Similarity = more results, less precise."
                    "</small>"
                )

                search_btn = gr.Button(
                    "Search", variant="secondary", size="lg", interactive=False
                )

        gr.Markdown("---")
        gr.Markdown("### Step 3 - Results")

        results_html = gr.HTML(
            value="<p style='color:#94a3b8;padding:8px'>Results will appear here after you search.</p>"
        )

        hit_selector = gr.Radio(
            choices=[],
            label="Select a result to generate and play that exact clip",
            visible=False,
            interactive=True,
        )

        seek_banner = gr.HTML(value="", visible=True)

        playback_html = gr.HTML(
            value="",
            visible=False,
        )

        gr.Markdown(
            """
        ---
        <div style='text-align:center;color:#94a3b8;font-size:13px;padding:8px 0'>
            Built with
            <a href='https://github.com/SYSTRAN/faster-whisper' style='color:#64748b'>faster-whisper</a> ·
            <a href='https://github.com/facebookresearch/faiss' style='color:#64748b'>FAISS</a> ·
            <a href='https://www.gradio.app' style='color:#64748b'>Gradio</a>
            &nbsp;·&nbsp;
            <a href='https://github.com/cksajil/omnisense' style='color:#64748b'>Source on GitHub</a>
        </div>
        """
        )

        # ── Event wiring ──────────────────────────────────────────────────────

        model_choice.change(
            fn=handle_model_change,
            inputs=[model_choice],
            outputs=[model_hint],
        )

        process_btn.click(
            fn=handle_process,
            inputs=[video_input, youtube_url, cookies_upload, model_choice],
            outputs=[status_md, search_btn],
        )

        search_btn.click(
            fn=handle_search,
            inputs=[query_box, top_k_slider, min_score_slider],
            outputs=[results_html, hit_selector],
        )

        hit_selector.change(
            fn=handle_hit_selected,
            inputs=[hit_selector],
            outputs=[playback_html, seek_banner],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[
                video_input,
                youtube_url,
                cookies_upload,
                model_choice,
                model_hint,
                status_md,
                search_btn,
                query_box,
                results_html,
                hit_selector,
                seek_banner,
                playback_html,
            ],
        )

    return demo


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # HF Spaces sets SPACE_ID automatically — it handles the public URL itself,
    # so share=True would redundantly try to tunnel through Gradio's servers.
    share = os.environ.get("SPACE_ID") is None

    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=share,
    )
