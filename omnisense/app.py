"""
omnisense/app.py

OmniSense — Temporal Video Search
Designed to run on HuggingFace Spaces CPU free tier (no GPU required).

Flow:
  1. User uploads video
  2. ffmpeg extracts 16kHz mono WAV
  3. faster-whisper transcribes → List[TranscriptChunk]
  4. MiniLM encodes chunks → FAISS index built
  5. User queries in natural language
  6. FAISS returns ranked hits with [start, end] timestamps
  7. User selects a hit → video seeks and plays from that timestamp
"""

from __future__ import annotations

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
# Gradio runs each event handler as a function call in the same process,
# so module-level state is safe for single-user local use and HF Spaces demos.
# For multi-user production, replace with gr.State() per session.

_index: TranscriptSearchIndex | None = None
_chunks: list[TranscriptChunk] = []
_video_path: str | None = None


# ── Event handlers ─────────────────────────────────────────────────────────────


def handle_process(
    video_file: str | None,
    model_size: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple[str, gr.update]:
    """
    Step 1: User uploads video and clicks Transcribe.
    Extracts audio, transcribes, builds search index.

    Returns:
        (status_markdown_string, search_button_update)
    """
    global _index, _chunks, _video_path

    if video_file is None:
        return "⚠️ Please upload a video file first.", gr.update(interactive=False)

    _video_path = video_file

    try:
        progress(0.05, desc="Extracting audio track…")
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = extract_audio(video_file, output_dir=tmpdir)

            progress(
                0.20,
                desc=(
                    f"Transcribing with Whisper [{model_size}] — "
                    f"this takes a few minutes on CPU, please wait…"
                ),
            )
            _chunks = transcribe(audio_path, model_size=model_size)

        if not _chunks:
            return (
                "⚠️ Transcription returned no segments. "
                "Check that the video has audible speech.",
                gr.update(interactive=False),
            )

        progress(0.85, desc="Building semantic search index…")
        _index = TranscriptSearchIndex()
        _index.build(_chunks)

        duration_s = _chunks[-1].end if _chunks else 0
        m, s = divmod(int(duration_s), 60)
        status = (
            f"✅ Ready! Indexed **{len(_chunks)} segments** "
            f"across **{m}m {s}s** of audio.\n\n"
            f"Type anything in the search box below to find it."
        )
        progress(1.0)
        return status, gr.update(interactive=True)

    except Exception as e:
        logger.exception("Error during processing")
        return f"❌ Error: {e}", gr.update(interactive=False)


def handle_search(
    query: str,
    top_k: int,
    min_score: float,
) -> tuple[str, gr.update]:
    """
    Step 2: User submits a search query.

    Returns:
        (results_html_string, hit_selector_update)
    """
    if _index is None or not _index.is_ready:
        return (
            "<p style='color:orange;padding:12px'>"
            "⚠️ Process a video first before searching.</p>",
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
            f'<strong>🔍 No matches found</strong> for <em>"{query}"</em><br>'
            "<small style='color:#666;margin-top:6px;display:block'>"
            "Try: rephrasing the query · lowering Min Similarity · "
            "using fewer / different keywords</small></div>"
        )
        return html, gr.update(visible=False)

    cards_html = _build_results_html(hits, query)
    radio_choices = [_hit_to_label(h) for h in hits]

    return cards_html, gr.update(
        choices=radio_choices,
        value=None,
        visible=True,
    )


def handle_hit_selected(label: str) -> gr.update:
    """
    Step 3: User clicks a search result radio button.
    Seeks the playback video to the hit's start timestamp.

    Returns:
        Gradio Video update with value= and time= set.
    """
    if not label or _video_path is None:
        return gr.update(visible=False)

    start_sec = _parse_start_from_label(label)
    logger.info(f"Seeking to {start_sec}s")

    return gr.update(
        value=_video_path,
        visible=True,
        time=start_sec,  # Gradio 4.x native video seek
    )


def handle_model_change(model_size: str) -> str:
    """Update the hint text when user changes model selection."""
    return f"ℹ️ **{model_size}** — {MODEL_SPEED_GUIDE.get(model_size, '')}"


def handle_clear() -> tuple:
    """Reset all state for a fresh start."""
    global _index, _chunks, _video_path
    _index = None
    _chunks = []
    _video_path = None
    return (
        None,  # video_input
        "base",  # model_choice
        f"ℹ️ **base** — {MODEL_SPEED_GUIDE['base']}",  # model_hint
        "👆 Upload a video and click **Transcribe & Index** to begin.",  # status_md
        gr.update(interactive=False),  # search_btn
        "",  # query_box
        "",  # results_html
        gr.update(choices=[], visible=False),  # hit_selector
        gr.update(visible=False),  # playback_video
    )


# ── Label helpers ──────────────────────────────────────────────────────────────


def _fmt_time(seconds: float) -> str:
    """Format seconds as M:SS string, e.g. 83.0 → '1:23'."""
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _hit_to_label(hit: SearchHit) -> str:
    """
    Encode a SearchHit as a Radio button label string.
    The start time is embedded so handle_hit_selected can parse it back.
    Format: '#1  [1:23 → 1:45]  score:72%  —  text preview…'
    """
    preview = hit.chunk.text[:100]
    if len(hit.chunk.text) > 100:
        preview += "…"
    return (
        f"#{hit.rank}  "
        f"[{_fmt_time(hit.chunk.start)} → {_fmt_time(hit.chunk.end)}]  "
        f"score:{hit.score:.0%}  —  {preview}"
    )


def _parse_start_from_label(label: str) -> float:
    """
    Extract start time in seconds from a hit label string.
    Parses the '[M:SS → …]' portion.
    """
    # label format: "#1  [1:23 → 1:45]  ..."
    time_str = label.split("[")[1].split("→")[0].strip()  # "1:23"
    parts = time_str.split(":")
    return int(parts[0]) * 60 + int(parts[1])


# ── Result card HTML ───────────────────────────────────────────────────────────


def _build_results_html(hits: list[SearchHit], query: str) -> str:
    """Render search results as styled HTML cards."""

    def _score_color(score: float) -> str:
        if score >= 0.60:
            return "#16a34a"  # green
        if score >= 0.40:
            return "#ca8a04"  # amber
        return "#dc2626"  # red

    header = (
        f"<div style='font-family:system-ui,-apple-system,sans-serif;padding:2px'>"
        f"<p style='color:#475569;font-size:14px;margin:0 0 14px 0'>"
        f"<strong>{len(hits)} result{'s' if len(hits) != 1 else ''}</strong> for "
        f"&ldquo;<em style='color:#2563eb'>{query}</em>&rdquo;"
        f"&nbsp;&nbsp;·&nbsp;&nbsp;"
        f"<span style='color:#94a3b8'>select a result below to jump to that moment ▼</span>"
        f"</p>"
    )

    cards = ""
    for h in hits:
        color = _score_color(h.score)
        bar_pct = int(h.score * 100)
        cards += f"""
        <div style="
            border:1px solid #e2e8f0;
            border-radius:10px;
            padding:14px 18px;
            margin-bottom:12px;
            background:#ffffff;
            box-shadow:0 1px 3px rgba(0,0,0,0.06);
        ">
            <div style="
                display:flex;
                justify-content:space-between;
                align-items:center;
                margin-bottom:8px;
            ">
                <span style="font-weight:700;font-size:15px;color:#1e293b">
                    #{h.rank}
                    <span style="
                        background:#1d4ed8;
                        color:#fff;
                        border-radius:5px;
                        padding:2px 10px;
                        font-size:13px;
                        margin-left:8px;
                        font-weight:600;
                        letter-spacing:0.2px;
                    ">
                        ⏱ {_fmt_time(h.chunk.start)} → {_fmt_time(h.chunk.end)}
                    </span>
                </span>
                <span style="font-size:13px;color:#64748b">
                    match &nbsp;
                    <strong style="color:{color};font-size:15px">{h.score:.0%}</strong>
                </span>
            </div>
            <div style="background:#f1f5f9;border-radius:4px;height:5px;margin-bottom:10px">
                <div style="
                    width:{bar_pct}%;
                    background:{color};
                    height:5px;
                    border-radius:4px;
                    transition:width 0.3s ease;
                "></div>
            </div>
            <p style="margin:0;color:#334155;font-size:14px;line-height:1.65">
                {h.chunk.text}
            </p>
        </div>
        """

    footer = "</div>"
    return header + cards + footer


# ── Gradio UI ──────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="OmniSense – Temporal Video Search",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
            .gradio-container { max-width: 1080px !important; }
            #model-radio label span { font-size: 13px !important; }
            footer { display: none !important; }
        """,
    ) as demo:
        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown(
            """
        # 🔍 OmniSense — Temporal Video Search
        **Find exactly *when* something was said in a video. No more scrubbing.**

        Upload any video → it gets transcribed → search in plain English
        → click a result → video jumps directly to that moment.

        > 🖥 Runs fully on CPU &nbsp;·&nbsp; No GPU required
        > &nbsp;·&nbsp; Powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + [FAISS](https://github.com/facebookresearch/faiss)
        """
        )

        # ── Top row: Upload + Search ────────────────────────────────────────────
        with gr.Row(equal_height=False):
            # Left: Upload + model picker
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### 📹 Step 1 — Upload & Transcribe")

                video_input = gr.Video(
                    label="Upload video",
                    height=240,
                )

                model_choice = gr.Radio(
                    choices=list(MODEL_SPEED_GUIDE.keys()),
                    value="base",
                    label="Whisper model  (speed ↔ accuracy)",
                    elem_id="model-radio",
                )
                model_hint = gr.Markdown(f"ℹ️ **base** — {MODEL_SPEED_GUIDE['base']}")

                with gr.Row():
                    process_btn = gr.Button(
                        "⚡ Transcribe & Index",
                        variant="primary",
                        size="lg",
                    )
                    clear_btn = gr.Button(
                        "🗑 Clear",
                        variant="secondary",
                        size="lg",
                    )

                status_md = gr.Markdown(
                    "👆 Upload a video and click **Transcribe & Index** to begin."
                )

            # Right: Search controls
            with gr.Column(scale=1, min_width=340):
                gr.Markdown("### 🔎 Step 2 — Search")

                query_box = gr.Textbox(
                    label="What are you looking for?",
                    placeholder=(
                        'e.g.  "climate change"  ·  '
                        '"when did he mention the budget?"  ·  '
                        '"machine learning"'
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
                    "Lower Min Similarity → more results, less precise. "
                    "Raise it to filter weak matches."
                    "</small>"
                )

                search_btn = gr.Button(
                    "🔍 Search",
                    variant="secondary",
                    size="lg",
                    interactive=False,  # enabled after transcription
                )

        # ── Results ─────────────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("### 📋 Step 3 — Results")

        results_html = gr.HTML(
            value="<p style='color:#94a3b8;padding:8px'>Results will appear here after you search.</p>"
        )

        hit_selector = gr.Radio(
            choices=[],
            label="▶  Select a result to jump to that moment in the video",
            visible=False,
            interactive=True,
        )

        # ── Playback ─────────────────────────────────────────────────────────────
        playback_video = gr.Video(
            label="▶  Playback — click a result above to seek here",
            visible=False,
            interactive=False,
            height=380,
        )

        # ── Footer ────────────────────────────────────────────────────────────────
        gr.Markdown(
            """
        ---
        <div style='text-align:center;color:#94a3b8;font-size:13px;padding:8px 0'>
            Built with ❤️ using
            <a href='https://github.com/SYSTRAN/faster-whisper' style='color:#64748b'>faster-whisper</a> ·
            <a href='https://github.com/facebookresearch/faiss' style='color:#64748b'>FAISS</a> ·
            <a href='https://www.gradio.app' style='color:#64748b'>Gradio</a>
            &nbsp;·&nbsp;
            <a href='https://github.com/cksajil/omnisense' style='color:#64748b'>Source on GitHub</a>
        </div>
        """
        )

        # ── Event wiring ──────────────────────────────────────────────────────────

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
            outputs=[results_html, hit_selector],
        )

        hit_selector.change(
            fn=handle_hit_selected,
            inputs=[hit_selector],
            outputs=[playback_video],
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
            ],
        )

    return demo


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",  # needed for HF Spaces / Docker
        server_port=7860,
        show_error=True,
        share=True
    )
