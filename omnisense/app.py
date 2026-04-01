"""
omnisense/app.py

OmniSense - Temporal Video Search
Upload a video, transcribe it, search in plain English, click a result to
preview that exact segment.
"""

from __future__ import annotations

# torch MUST be the first heavy import to register OpenMP before ctranslate2.
# Prevents segfault on macOS. Do not move this import.
import torch  # noqa: F401

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


# ── Module-level session state ─────────────────────────────────────────────────

_index: TranscriptSearchIndex | None = None
_chunks: list[TranscriptChunk] = []
_video_path: str | None = None
_preview_dir: str = tempfile.mkdtemp(prefix="omnisense_preview_")
_last_clip_path: str | None = None


# ── Event handlers ─────────────────────────────────────────────────────────────


def handle_process(
    video_file: str | None,
    model_size: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, gr.update]:
    global _index, _chunks, _video_path, _last_clip_path

    _last_clip_path = None

    if video_file is None:
        return "Upload a video file first.", gr.update(interactive=False)

    _video_path = video_file

    try:
        progress(0.15, desc="Extracting audio track...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = extract_audio(video_file, output_dir=tmpdir)
            progress(0.35, desc=f"Transcribing [{model_size}] — this may take a minute...")
            _chunks = transcribe(audio_path, model_size=model_size)

        if not _chunks:
            return (
                "No speech found. Check that the video has audible audio.",
                gr.update(interactive=False),
            )

        progress(0.85, desc="Building search index...")
        _index = TranscriptSearchIndex()
        _index.build(_chunks)

        duration_s = _chunks[-1].end if _chunks else 0
        m, s = divmod(int(duration_s), 60)
        progress(1.0)
        return (
            f"Ready — {len(_chunks)} segments indexed ({m}m {s}s). Enter a search query.",
            gr.update(interactive=True),
        )

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
            "<p style='color:orange;padding:8px'>Transcribe a video first.</p>",
            gr.update(visible=False),
        )

    query = query.strip()
    if not query:
        return (
            "<p style='color:#888;padding:8px'>Enter a search query.</p>",
            gr.update(visible=False),
        )

    hits = _index.search(query, top_k=int(top_k), min_score=float(min_score))

    if not hits:
        return (
            "<div style='padding:12px;border-radius:8px;background:#fff8e1;"
            "border:1px solid #ffe082'>"
            f"No matches for <em>\"{query}\"</em>. "
            "Try rephrasing or lowering Min Similarity.</div>",
            gr.update(visible=False),
        )

    cards_html = _build_results_html(hits, query)
    radio_choices = [_hit_to_label(h) for h in hits]
    return (
        cards_html,
        gr.update(choices=radio_choices, value=None, visible=True),
    )


def handle_hit_selected(label: str) -> tuple[gr.update, gr.update]:
    if not label or _video_path is None:
        return gr.update(value=None, visible=False), gr.update(value="")

    start_sec, end_sec = _parse_times_from_label(label)
    logger.info(f"Selected hit: {start_sec}s -> {end_sec}s")

    try:
        clip_path = _create_preview_clip(_video_path, start_sec, end_sec)
        return gr.update(value=clip_path, visible=True), gr.update(value="")

    except Exception as e:
        logger.exception("Error while creating preview clip")
        return (
            gr.update(value=None, visible=False),
            gr.update(value=f"<p style='color:#991b1b'>Preview failed: {e}</p>"),
        )


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
        None,                                   # video_input
        "base",                                 # model_choice
        "Upload a video to get started.",       # status_md
        gr.update(interactive=False),           # search_btn
        "",                                     # query_box
        "",                                     # results_html
        gr.update(choices=[], visible=False),   # hit_selector
        gr.update(value=None, visible=False),   # playback_video
    )


# ── Clip helpers ───────────────────────────────────────────────────────────────


def _create_preview_clip(source_path: str, start_sec: float, end_sec: float) -> str:
    global _last_clip_path, _preview_dir

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video file not found: {source_path}")

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec + 0.2, float(end_sec))
    duration = end_sec - start_sec

    clip_name = f"clip_{uuid.uuid4().hex}.mp4"
    clip_path = os.path.join(_preview_dir, clip_name)

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", source_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
        clip_path,
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0 or not os.path.isfile(clip_path):
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-1200:]}")

    _last_clip_path = clip_path
    return clip_path


# ── Label / time helpers ───────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _hit_to_label(hit: SearchHit) -> str:
    preview = hit.chunk.text[:80]
    if len(hit.chunk.text) > 80:
        preview += "..."
    return (
        f"#{hit.rank}  [{_fmt_time(hit.chunk.start)} → {_fmt_time(hit.chunk.end)}]"
        f"  {hit.score:.0%}  —  {preview}"
    )


def _parse_times_from_label(label: str) -> tuple[float, float]:
    range_text = label.split("[")[1].split("]")[0]
    start_str, end_str = [x.strip() for x in range_text.split("→")]

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

    cards = "<div style='font-family:system-ui,-apple-system,sans-serif'>"
    for h in hits:
        color = _score_color(h.score)
        cards += f"""
        <div style="border:1px solid #e2e8f0;border-radius:8px;
                    padding:12px 16px;margin-bottom:10px;background:#fff;">
            <div style="display:flex;justify-content:space-between;
                        align-items:center;margin-bottom:6px;">
                <span style="font-weight:700;color:#1e293b">
                    <span style="background:#1d4ed8;color:#fff;border-radius:4px;
                                 padding:2px 8px;font-size:12px;">
                        {_fmt_time(h.chunk.start)} – {_fmt_time(h.chunk.end)}
                    </span>
                </span>
                <span style="font-size:13px;color:{color};font-weight:600">
                    {h.score:.0%} match
                </span>
            </div>
            <p style="margin:0;color:#334155;font-size:14px;line-height:1.6">
                {h.chunk.text}
            </p>
        </div>
        """
    cards += "</div>"
    return cards


# ── Gradio UI ──────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="OmniSense",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
            .gradio-container { max-width: 900px !important; }
            footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown(
            "# OmniSense\n"
            "Upload a video, transcribe it, then search for any moment by what was said."
        )

        with gr.Row(equal_height=False):

            # ── Left: upload + transcribe ──────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                video_input = gr.Video(label="Video", height=220)

                model_choice = gr.Radio(
                    choices=list(MODEL_SPEED_GUIDE.keys()),
                    value="base",
                    label="Whisper model",
                )

                with gr.Row():
                    process_btn = gr.Button("Transcribe", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")

                status_md = gr.Markdown("Upload a video to get started.")

            # ── Right: search ──────────────────────────────────────────────
            with gr.Column(scale=1, min_width=300):
                query_box = gr.Textbox(
                    label="Search",
                    placeholder='e.g. "climate change" or "budget announcement"',
                    lines=2,
                )

                with gr.Row():
                    top_k_slider = gr.Slider(
                        minimum=1, maximum=10, value=5, step=1, label="Max results"
                    )
                    min_score_slider = gr.Slider(
                        minimum=0.10, maximum=0.90, value=0.30, step=0.05,
                        label="Min similarity"
                    )

                search_btn = gr.Button("Search", variant="primary", interactive=False)

        gr.Markdown("---")

        results_html = gr.HTML(
            value="<p style='color:#94a3b8'>Search results will appear here.</p>"
        )

        hit_selector = gr.Radio(
            choices=[],
            label="Select a result to play that segment",
            visible=False,
            interactive=True,
        )

        error_banner = gr.HTML(value="", visible=True)

        playback_video = gr.Video(
            label="Segment preview",
            visible=False,
            autoplay=True,
        )

        # ── Event wiring ──────────────────────────────────────────────────

        process_btn.click(
            fn=handle_process,
            inputs=[video_input, model_choice],
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
            outputs=[playback_video, error_banner],
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[
                video_input,
                model_choice,
                status_md,
                search_btn,
                query_box,
                results_html,
                hit_selector,
                playback_video,
            ],
        )

    return demo


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # HF Spaces sets SPACE_ID automatically — it handles the public URL itself.
    share = os.environ.get("SPACE_ID") is None

    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=share,
    )
