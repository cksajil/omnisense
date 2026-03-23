"""
OmniSense Gradio Dashboard.

Single-file interactive UI that chains all four pipelines:
    Audio → NLP → Vision → Search

Users upload a video or audio file, the app runs all pipelines
in sequence and displays results in a tabbed interface with
a live semantic search bar at the bottom.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import gradio as gr

from omnisense.config import DEVICE
from omnisense.pipelines.audio import AudioPipeline
from omnisense.pipelines.nlp import NLPPipeline
from omnisense.pipelines.search import SearchPipeline
from omnisense.pipelines.vision import VisionPipeline
from omnisense.utils.logger import log

# ── Initialise pipelines once at startup ──────────────────────────────────────
log.info("Initialising pipelines…")
audio_pipeline = AudioPipeline(device=DEVICE)
nlp_pipeline = NLPPipeline(device=DEVICE)
vision_pipeline = VisionPipeline(device=DEVICE)
search_pipeline = SearchPipeline(device=DEVICE)
log.info("All pipelines ready ✓")

# ── Global state ──────────────────────────────────────────────────────────────
_last_results: dict = {}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


# ── Main analysis function ────────────────────────────────────────────────────


def analyse_media(
    media_file: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Run all four pipelines on the uploaded file.
    Returns a tuple matching the Gradio output component order.
    """
    global _last_results

    if media_file is None:
        msg = "⚠️ Please upload a file first."
        return msg, msg, msg, msg, msg, gr.update(interactive=False)

    try:
        media_path = Path(media_file)
        log.info(f"Starting analysis: {media_path.name}")

        # ── 1. Audio ──────────────────────────────────────────────────────────
        progress(0.05, desc="🎙 Loading audio model…")
        log.info("[1/4] Audio pipeline starting…")
        audio_result = audio_pipeline(media_path)
        log.info(
            f"[1/4] Audio done — "
            f"{len(audio_result['segments'])} segments, "
            f"language={audio_result['language']}"
        )

        # ── 2. NLP ────────────────────────────────────────────────────────────
        progress(0.30, desc="🧠 Running NLP analysis…")
        log.info("[2/4] NLP pipeline starting…")
        nlp_result = nlp_pipeline(audio_result)
        log.info(f"[2/4] NLP done — top topic: {nlp_result['top_topic']}")

        # ── 3. Vision ─────────────────────────────────────────────────────────
        vision_result = None
        if media_path.suffix.lower() in VIDEO_EXTENSIONS:
            progress(0.60, desc="🖼 Analysing video frames…")
            log.info("[3/4] Vision pipeline starting…")
            vision_result = vision_pipeline(media_path, max_frames=10)
            log.info(
                f"[3/4] Vision done — "
                f"{vision_result['frame_count']} frames, "
                f"top label: {vision_result['top_visual_label']}"
            )
        else:
            log.info("[3/4] Vision skipped — audio-only file")

        # ── 4. Search index ───────────────────────────────────────────────────
        progress(0.85, desc="🔍 Building search index…")
        log.info("[4/4] Building search index…")
        search_pipeline.build_index(
            audio_result=audio_result,
            nlp_result=nlp_result,
            vision_result=vision_result,
        )
        log.info(
            f"[4/4] Index built — {search_pipeline.get_stats()['document_count']} docs"
        )

        _last_results = {
            "audio": audio_result,
            "nlp": nlp_result,
            "vision": vision_result,
        }

        progress(1.0, desc="✅ Analysis complete!")
        log.info("Analysis complete ✓")

        return (
            _format_overview(audio_result, nlp_result, vision_result),
            _format_transcript(audio_result),
            _format_nlp(nlp_result),
            _format_vision(vision_result),
            _format_search_ready(search_pipeline.get_stats()),
            gr.update(interactive=True),
        )

    except Exception:
        error_detail = traceback.format_exc()
        log.error(f"Analysis failed:\n{error_detail}")
        error_md = f"## ❌ Error\n\n```\n{error_detail}\n```"
        return (
            error_md,
            error_md,
            error_md,
            error_md,
            error_md,
            gr.update(interactive=False),
        )


# ── Search function ───────────────────────────────────────────────────────────


def semantic_search(query: str, top_k: int = 5) -> str:
    """Run semantic search over the last analysis results."""
    if not query.strip():
        return "Enter a query above to search."

    if not search_pipeline._index_built:
        return "⚠️ Run analysis first, then search."

    results = search_pipeline.query(query, top_k=int(top_k))

    if not results:
        return "No results found."

    lines = [f"### Results for: *{query}*\n"]
    for i, r in enumerate(results, 1):
        score_pct = int(r["score"] * 20)
        score_bar = "█" * max(score_pct, 1)
        lines.append(
            f"**{i}. [{r['source'].upper()}]** — "
            f"score: `{r['score']:.4f}` {score_bar}"
        )
        lines.append(f"> {r['text'][:250]}")
        if r["metadata"]:
            meta_parts = [
                f"{k}: {v}"
                for k, v in r["metadata"].items()
                if k != "words" and v != ""
            ]
            if meta_parts:
                lines.append(f"*{' | '.join(meta_parts)}*")
        lines.append("")

    return "\n".join(lines)


# ── Output formatters ─────────────────────────────────────────────────────────


def _format_overview(
    audio: dict,
    nlp: dict,
    vision: dict | None,
) -> str:
    lines = ["## 📊 Analysis Overview\n"]
    lines.append(f"**Language detected:** {audio.get('language', 'unknown').upper()}")
    lines.append(f"**Duration:** {audio.get('duration', 0):.1f}s")
    lines.append(f"**Word count:** {nlp.get('word_count', 0):,}")
    lines.append(f"**Top topic:** {nlp.get('top_topic', 'unknown').title()}")

    if vision:
        lines.append(f"**Frames analysed:** {vision.get('frame_count', 0)}")
        lines.append(
            f"**Visual scene:** " f"{vision.get('top_visual_label', 'unknown').title()}"
        )
        objs = vision.get("unique_objects", [])
        if objs:
            lines.append(f"**Objects detected:** {', '.join(objs[:8])}")

    lines.append("\n### Topic Distribution\n")
    for t in nlp.get("topics", [])[:5]:
        bar = "█" * max(int(t["score"] * 30), 1)
        lines.append(f"`{t['label']:<20}` {bar} `{t['score']:.3f}`")

    entities = nlp.get("entities", [])
    if entities:
        lines.append("\n### Key Entities\n")
        for ent in entities[:8]:
            lines.append(f"- **{ent.get('text', '')}** " f"[{ent.get('label', '')}]")

    return "\n".join(lines)


def _format_transcript(audio: dict) -> str:
    lines = ["## 📝 Transcript\n"]
    segments = audio.get("segments", [])

    if not segments:
        text = audio.get("transcript", "No transcript available.")
        lines.append(text)
        return "\n".join(lines)

    for seg in segments:
        start = seg.get("start", 0)
        mm, ss = divmod(int(start), 60)
        text = seg.get("text", "").strip()
        if text:
            lines.append(f"**[{mm:02d}:{ss:02d}]** {text}")

    lines.append(
        f"\n---\n*{len(segments)} segments · "
        f"language: {audio.get('language', '?').upper()}*"
    )
    return "\n".join(lines)


def _format_nlp(nlp: dict) -> str:
    lines = ["## 🧠 NLP Analysis\n"]

    lines.append("### Summary\n")
    lines.append(nlp.get("summary", "No summary available."))

    lines.append("\n### Named Entities\n")
    entities = nlp.get("entities", [])
    if entities:
        for ent in entities[:12]:
            lines.append(
                f"- **{ent.get('text', '')}** "
                f"[{ent.get('label', '')}] "
                f"— confidence: `{ent.get('score', 0):.2f}`"
            )
    else:
        lines.append("No entities detected.")

    lines.append("\n### Topic Classification\n")
    for t in nlp.get("topics", [])[:8]:
        bar = "█" * max(int(t["score"] * 30), 1)
        lines.append(f"`{t['label']:<20}` {bar} `{t['score']:.3f}`")

    return "\n".join(lines)


def _format_vision(vision: dict | None) -> str:
    if vision is None:
        return (
            "## 🖼 Vision Analysis\n\n"
            "Upload a **video file** to enable vision analysis.\n\n"
            "*Audio-only files skip this pipeline.*"
        )

    lines = ["## 🖼 Vision Analysis\n"]
    lines.append(f"**Frames processed:** {vision.get('frame_count', 0)}")
    lines.append(
        f"**Top visual label:** "
        f"{vision.get('top_visual_label', 'unknown').title()}\n"
    )

    lines.append("### Frame Captions\n")
    for cap in vision.get("captions", [])[:10]:
        ts = cap.get("timestamp", 0)
        mm, ss = divmod(int(ts), 60)
        caption = cap.get("caption", "")
        if caption and caption != "caption unavailable":
            lines.append(f"**[{mm:02d}:{ss:02d}]** {caption}")

    lines.append("\n### Detected Objects\n")
    objs = vision.get("unique_objects", [])
    if objs:
        lines.append(", ".join(f"`{o}`" for o in objs))
    else:
        lines.append("No objects detected above confidence threshold.")

    lines.append("\n### CLIP Scene Classification\n")
    for c in vision.get("clip_labels", [])[:6]:
        bar = "█" * max(int(c["score"] * 30), 1)
        lines.append(f"`{c['label']:<30}` {bar} `{c['score']:.3f}`")

    return "\n".join(lines)


def _format_search_ready(stats: dict) -> str:
    lines = ["## 🔍 Search Index\n"]
    status = stats.get("status", "unknown")
    emoji = "✅" if status == "ready" else "⏳"
    lines.append(f"**Status:** {emoji} {status.title()}")
    lines.append(f"**Documents indexed:** {stats.get('document_count', 0):,}")
    lines.append(f"**Embedding dimensions:** {stats.get('embedding_dim', 384)}")

    sources = stats.get("sources", [])
    if sources:
        lines.append("\n**Indexed sources:**")
        source_descriptions = {
            "transcript": "Individual transcript segments with timestamps",
            "transcript_chunk": "Longer transcript chunks for context",
            "summary": "Full document summary",
            "chunk_summary": "Per-chunk summaries",
            "entity": "Named entities (people, orgs, locations)",
            "caption": "Video frame captions with timestamps",
            "objects": "Detected objects across all frames",
        }
        for src in sorted(sources):
            desc = source_descriptions.get(src, src)
            lines.append(f"- **{src}** — {desc}")

    lines.append(
        "\n*Use the search bar below to query across all sources "
        "using natural language.*"
    )
    return "\n".join(lines)


# ── Gradio UI ──────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="OmniSense — Multimodal AI Media Analyzer",
        theme=gr.themes.Soft(),
        css="""
            .header { text-align: center; padding: 1.5rem 0 0.5rem; }
            .header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
            .header p  { color: #666; font-size: 1rem; margin: 0.25rem 0 0; }
            footer { display: none !important; }
        """,
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(
            """
            <div class="header">
                <h1>🎬 OmniSense</h1>
                <p>
                    Multimodal AI Media Analyzer &nbsp;·&nbsp;
                    Transcription &nbsp;·&nbsp; NLP &nbsp;·&nbsp;
                    Vision &nbsp;·&nbsp; Semantic Search
                </p>
            </div>
        """
        )

        # ── Upload row ────────────────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Column(scale=2):
                media_input = gr.File(
                    label="📁 Upload Media File",
                    file_types=["video", "audio"],
                    type="filepath",
                )
                analyse_btn = gr.Button(
                    "🚀  Analyse",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown(
                    "*Supported: MP4, MOV, AVI, MKV, MP3, WAV, M4A, FLAC*",
                )
            with gr.Column(scale=3):
                overview_out = gr.Markdown(
                    value=(
                        "### Welcome to OmniSense 👋\n\n"
                        "Upload a video or audio file and click **Analyse** "
                        "to run the full pipeline:\n\n"
                        "1. 🎙 **Audio** — Whisper transcription with timestamps\n"
                        "2. 🧠 **NLP** — Summarisation, NER, topic classification\n"
                        "3. 🖼 **Vision** — Frame captioning and object detection\n"
                        "4. 🔍 **Search** — Semantic index over all outputs\n"
                    ),
                )

        # ── Results tabs ──────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("📝 Transcript"):
                transcript_out = gr.Markdown(
                    value="*Results will appear here after analysis.*"
                )
            with gr.Tab("🧠 NLP Analysis"):
                nlp_out = gr.Markdown(
                    value="*Results will appear here after analysis.*"
                )
            with gr.Tab("🖼 Vision Analysis"):
                vision_out = gr.Markdown(
                    value="*Results will appear here after analysis.*"
                )
            with gr.Tab("🔍 Search Index"):
                search_index_out = gr.Markdown(
                    value="*Results will appear here after analysis.*"
                )

        # ── Semantic search ───────────────────────────────────────────────────
        gr.Markdown("---\n## 🔍 Semantic Search")
        gr.Markdown(
            "After analysis, search across transcript, summary, "
            "entities, and captions using natural language."
        )

        with gr.Row():
            search_input = gr.Textbox(
                placeholder="e.g. what did the speaker say about technology?",
                label="Search Query",
                scale=4,
                lines=1,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Results (Top K)",
                scale=1,
            )

        search_btn = gr.Button(
            "🔍  Search",
            variant="secondary",
            interactive=False,
        )
        search_out = gr.Markdown(value="*Run analysis first, then search.*")

        gr.Examples(
            examples=[
                ["what did the speaker talk about?"],
                ["what objects were detected in the video?"],
                ["who are the people or organisations mentioned?"],
                ["describe the visual scenes"],
                ["what is the main topic or theme?"],
                ["any technology or science mentioned?"],
            ],
            inputs=search_input,
            label="Example queries",
        )

        # ── Footer ────────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "*Built with 🤗 HuggingFace · "
            "faster-whisper · BART · BERT-NER · CLIP · BLIP · DETR · FAISS*"
        )

        # ── Event wiring ──────────────────────────────────────────────────────
        analyse_btn.click(
            fn=analyse_media,
            inputs=[media_input],
            outputs=[
                overview_out,
                transcript_out,
                nlp_out,
                vision_out,
                search_index_out,
                search_btn,
            ],
            show_progress="full",
        )

        search_btn.click(
            fn=semantic_search,
            inputs=[search_input, top_k_slider],
            outputs=[search_out],
        )

        search_input.submit(
            fn=semantic_search,
            inputs=[search_input, top_k_slider],
            outputs=[search_out],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
