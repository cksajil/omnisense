"""
omnisense/app.py

OmniSense - Temporal Video Search
Designed to run on HuggingFace Spaces CPU free tier (no GPU required).

Flow:
  1. User uploads video
  2. ffmpeg extracts 16kHz mono WAV
  3. faster-whisper transcribes -> List[TranscriptChunk]
  4. MiniLM encodes chunks -> FAISS index built
  5. User queries in natural language
  6. FAISS returns ranked hits with [start, end] timestamps
  7. User selects a hit -> JS seeks the video to that timestamp

Import order note:
  torch is imported first to register OpenMP before ctranslate2 (faster-whisper)
  loads. On macOS this prevents a segfault. Do not move or remove this import.
"""

from __future__ import annotations

# torch MUST be the first heavy import — registers OpenMP before ctranslate2
import torch  # noqa: F401

import tempfile

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


# ── Event handlers ─────────────────────────────────────────────────────────────

def handle_process(
    video_file: str | None,
    model_size: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, gr.update]:
    global _index, _chunks, _video_path

    if video_file is None:
        return "Please upload a video file first.", gr.update(interactive=False)

    _video_path = video_file

    try:
        progress(0.05, desc="Extracting audio track...")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = extract_audio(video_file, output_dir=tmpdir)

            progress(
                0.20,
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
) -> tuple[str, gr.update, gr.update]:
    if _index is None or not _index.is_ready:
        return (
            "<p style='color:orange;padding:12px'>Process a video first.</p>",
            gr.update(visible=False),
            gr.update(value=""),
        )

    query = query.strip()
    if not query:
        return (
            "<p style='color:#888;padding:12px'>Enter a search query above.</p>",
            gr.update(visible=False),
            gr.update(value=""),
        )

    hits = _index.search(query, top_k=int(top_k), min_score=float(min_score))

    if not hits:
        html = (
            "<div style='padding:16px;border-radius:8px;"
            "background:#fff8e1;border:1px solid #ffe082'>"
            f"<strong>No matches found</strong> for <em>\"{query}\"</em><br>"
            "<small style='color:#666;margin-top:6px;display:block'>"
            "Try rephrasing, or lower the Min Similarity slider."
            "</small></div>"
        )
        return html, gr.update(visible=False), gr.update(value="")

    cards_html = _build_results_html(hits, query)
    radio_choices = [_hit_to_label(h) for h in hits]

    return (
        cards_html,
        gr.update(choices=radio_choices, value=None, visible=True),
        gr.update(value=""),
    )


def handle_hit_selected(label: str) -> tuple[gr.update, gr.update, str]:
    """
    User selects a search hit.
    Returns:
      - playback_video: show the video file
      - seek_box: hidden number box with the start time in seconds
      - seek_js trigger value: timestamp string to fire the JS seek
    """
    if not label or _video_path is None:
        return gr.update(visible=False), gr.update(value=0), ""

    start_sec = _parse_start_from_label(label)
    logger.info(f"Seeking to {start_sec}s")

    return (
        gr.update(value=_video_path, visible=True),
        gr.update(value=start_sec),
        str(start_sec),   # triggers the JS seek via Textbox change
    )


def handle_model_change(model_size: str) -> str:
    return f"**{model_size}** -- {MODEL_SPEED_GUIDE.get(model_size, '')}"


def handle_clear() -> tuple:
    global _index, _chunks, _video_path
    _index = None
    _chunks = []
    _video_path = None
    return (
        None,
        "base",
        f"**base** -- {MODEL_SPEED_GUIDE['base']}",
        "Upload a video and click Transcribe and Index to begin.",
        gr.update(interactive=False),
        "",
        "",
        gr.update(choices=[], visible=False),
        gr.update(visible=False),
        gr.update(value=0),
        "",
    )


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


def _parse_start_from_label(label: str) -> float:
    # label: "#1  [1:23 -> 1:45]  score:72%  --  ..."
    time_str = label.split("[")[1].split("->")[0].strip()
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


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
        "<span style='color:#94a3b8'>select a result below to jump to that moment</span>"
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
                            border-radius:4px;transition:width 0.3s ease;"></div>
            </div>
            <p style="margin:0;color:#334155;font-size:14px;line-height:1.65">
                {h.chunk.text}
            </p>
        </div>
        """

    return header + cards + "</div>"


# ── JS for video seek ──────────────────────────────────────────────────────────
# Gradio does not support gr.Video(time=...) in all versions.
# Instead we use a hidden Number component to carry the seek time,
# and a JS snippet wired to its change event to seek the <video> element.

SEEK_JS = """
(seek_seconds) => {
    // Find the playback video element — it's the second gr.Video on the page
    const videos = document.querySelectorAll('video');
    if (videos.length === 0) return seek_seconds;
    // Use the last video element (the playback one, not the upload one)
    const vid = videos[videos.length - 1];
    if (vid) {
        vid.currentTime = seek_seconds;
        vid.play();
    }
    return seek_seconds;
}
"""


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

        gr.Markdown("""
        # OmniSense - Temporal Video Search
        **Find exactly *when* something was said in a video. No more scrubbing.**

        Upload a video, transcribe it, search in plain English,
        then click a result to jump directly to that moment.

        > Runs fully on CPU · No GPU required
        > · Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + [FAISS](https://github.com/facebookresearch/faiss)
        """)

        with gr.Row(equal_height=False):

            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### Step 1 - Upload and Transcribe")

                video_input = gr.Video(label="Upload video", height=240)

                model_choice = gr.Radio(
                    choices=list(MODEL_SPEED_GUIDE.keys()),
                    value="base",
                    label="Whisper model (speed vs accuracy)",
                    elem_id="model-radio",
                )
                model_hint = gr.Markdown(
                    f"**base** -- {MODEL_SPEED_GUIDE['base']}"
                )

                with gr.Row():
                    process_btn = gr.Button(
                        "Transcribe and Index", variant="primary", size="lg"
                    )
                    clear_btn = gr.Button(
                        "Clear", variant="secondary", size="lg"
                    )

                status_md = gr.Markdown(
                    "Upload a video and click Transcribe and Index to begin."
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
                        minimum=1, maximum=10, value=5, step=1,
                        label="Max results",
                    )
                    min_score_slider = gr.Slider(
                        minimum=0.10, maximum=0.90, value=0.30, step=0.05,
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
            label="Select a result to jump to that moment in the video",
            visible=False,
            interactive=True,
        )

        # Hidden number box carries seek time to the JS function
        seek_time = gr.Number(value=0, visible=False)
        # Hidden textbox change fires the JS seek trigger
        seek_trigger = gr.Textbox(value="", visible=False)

        playback_video = gr.Video(
            label="Playback - select a result above to seek here",
            visible=False,
            interactive=False,
            height=380,
        )

        gr.Markdown("""
        ---
        <div style='text-align:center;color:#94a3b8;font-size:13px;padding:8px 0'>
            Built with
            <a href='https://github.com/SYSTRAN/faster-whisper' style='color:#64748b'>faster-whisper</a> ·
            <a href='https://github.com/facebookresearch/faiss' style='color:#64748b'>FAISS</a> ·
            <a href='https://www.gradio.app' style='color:#64748b'>Gradio</a>
            &nbsp;·&nbsp;
            <a href='https://github.com/cksajil/omnisense' style='color:#64748b'>Source on GitHub</a>
        </div>
        """)

        # ── Event wiring ──────────────────────────────────────────────────────

        model_choice.change(
            fn=handle_model_change,
            inputs=[model_choice],
            outputs=[model_hint],
        )

        process_btn.click(
            fn=handle_process,
            inputs=[video_input, model_choice],
            outputs=[status_md, search_btn],
        )

        search_btn.click(
            fn=handle_search,
            inputs=[query_box, top_k_slider, min_score_slider],
            outputs=[results_html, hit_selector, seek_trigger],
        )

        hit_selector.change(
            fn=handle_hit_selected,
            inputs=[hit_selector],
            outputs=[playback_video, seek_time, seek_trigger],
        )

        # JS seek: fires when seek_trigger changes, reads seek_time, seeks video
        seek_trigger.change(
            fn=None,
            inputs=[seek_time],
            outputs=[seek_time],
            js=SEEK_JS,
        )

        clear_btn.click(
            fn=handle_clear,
            inputs=[],
            outputs=[
                video_input,
                model_choice,
                model_hint,
                status_md,
                search_btn,
                query_box,
                results_html,
                hit_selector,
                playback_video,
                seek_time,
                seek_trigger,
            ],
        )

    return demo


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        share=True
    )