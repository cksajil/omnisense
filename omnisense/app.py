"""
OmniSense Gradio Dashboard.

Single-file interactive UI that chains all four pipelines:
    Audio → NLP → Vision → Search

Users upload a video or audio file, the app runs all pipelines
in sequence and displays results in a tabbed interface with
a live semantic search bar at the bottom.
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from omnisense.config import DEVICE
from omnisense.pipelines.audio import AudioPipeline
from omnisense.pipelines.nlp import NLPPipeline
from omnisense.pipelines.search import SearchPipeline
from omnisense.pipelines.vision import VisionPipeline
from omnisense.utils.logger import log

# ── Initialise pipelines once at startup (model weights cached) ───────────────
log.info("Initialising pipelines…")
audio_pipeline = AudioPipeline(device=DEVICE)
nlp_pipeline = NLPPipeline(device=DEVICE)
vision_pipeline = VisionPipeline(device=DEVICE)
search_pipeline = SearchPipeline(device=DEVICE)
log.info("All pipelines ready ✓")

# ── Global state — holds last analysis results for search ─────────────────────
_last_results: dict = {}


def analyse_media(media_file: str) -> tuple:
    """
    Main analysis function — runs all four pipelines on uploaded file.

    Args:
        media_file: Path to uploaded file from Gradio.

    Returns:
        Tuple of outputs matching Gradio component order.
    """
    global _last_results

    if media_file is None:
        empty = "Upload a file to begin."
        return empty, empty, empty, empty, empty, gr.update(interactive=False)

    try:
        media_path = Path(media_file)
        log.info(f"Analysing: {media_path.name}")

        # ── 1. Audio ─────────────────────────────────────────────────────────
        log.info("Running audio pipeline…")
        audio_result = audio_pipeline(media_path)

        # ── 2. NLP ───────────────────────────────────────────────────────────
        log.info("Running NLP pipeline…")
        nlp_result = nlp_pipeline(audio_result)

        # ── 3. Vision ────────────────────────────────────────────────────────
        vision_result = None
        suffix = media_path.suffix.lower()
        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
        if suffix in video_exts:
            log.info("Running vision pipeline…")
            vision_result = vision_pipeline(media_path, max_frames=10)

        # ── 4. Search index ───────────────────────────────────────────────────
        log.info("Building search index…")
        search_pipeline.build_index(
            audio_result=audio_result,
            nlp_result=nlp_result,
            vision_result=vision_result,
        )

        _last_results = {
            "audio": audio_result,
            "nlp": nlp_result,
            "vision": vision_result,
        }

        # ── Format outputs ────────────────────────────────────────────────────
        transcript_md = _format_transcript(audio_result)
        nlp_md = _format_nlp(nlp_result)
        vision_md = _format_vision(vision_result)
        search_md = _format_search_ready(search_pipeline.get_stats())
        overview_md = _format_overview(audio_result, nlp_result, vision_result)

        log.info("Analysis complete ✓")
        return (
            overview_md,
            transcript_md,
            nlp_md,
            vision_md,
            search_md,
            gr.update(interactive=True),
        )

    except Exception as exc:
        log.error(f"Analysis failed: {exc}")
        error_md = f"## Error\n```\n{exc}\n```"
        return (
            error_md,
            error_md,
            error_md,
            error_md,
            error_md,
            gr.update(interactive=False),
        )


def semantic_search(query: str, top_k: int = 5) -> str:
    """Run semantic search over the last analysis results."""
    if not query.strip():
        return "Enter a query above to search."

    if not search_pipeline._index_built:
        return "Run analysis first, then search."

    results = search_pipeline.query(query, top_k=int(top_k))

    if not results:
        return "No results found."

    lines = [f"### Results for: *{query}*\n"]
    for i, r in enumerate(results, 1):
        score_bar = "█" * int(r["score"] * 20)
        lines.append(
            f"**{i}. [{r['source'].upper()}]** — score: `{r['score']:.4f}` {score_bar}"
        )
        lines.append(f"> {r['text'][:200]}")
        if r["metadata"]:
            meta = " | ".join(
                f"{k}: {v}" for k, v in r["metadata"].items() if k not in ("words",)
            )
            lines.append(f"*{meta}*")
        lines.append("")

    return "\n".join(lines)


# ── Formatters ────────────────────────────────────────────────────────────────


def _format_overview(audio: dict, nlp: dict, vision: dict | None) -> str:
    lines = ["## Analysis Overview\n"]
    lines.append(f"**Language:** {audio.get('language', 'unknown').upper()}")
    lines.append(f"**Duration:** {audio.get('duration', 0):.1f}s")
    lines.append(f"**Word count:** {nlp.get('word_count', 0)}")
    lines.append(f"**Top topic:** {nlp.get('top_topic', 'unknown').title()}")
    if vision:
        lines.append(f"**Frames analysed:** {vision.get('frame_count', 0)}")
        lines.append(
            f"**Visual scene:** {vision.get('top_visual_label', 'unknown').title()}"
        )
        objs = vision.get("unique_objects", [])
        if objs:
            lines.append(f"**Objects detected:** {', '.join(objs[:8])}")
    lines.append("\n### Topic Distribution")
    for t in nlp.get("topics", [])[:5]:
        bar = "█" * int(t["score"] * 30)
        lines.append(f"`{t['label']:<20}` {bar} {t['score']:.3f}")
    return "\n".join(lines)


def _format_transcript(audio: dict) -> str:
    lines = ["## Transcript\n"]
    segments = audio.get("segments", [])
    if not segments:
        lines.append(audio.get("transcript", "No transcript available."))
        return "\n".join(lines)
    for seg in segments:
        start = seg.get("start", 0)
        mm, ss = divmod(int(start), 60)
        lines.append(f"**[{mm:02d}:{ss:02d}]** {seg.get('text', '').strip()}")
    return "\n".join(lines)


def _format_nlp(nlp: dict) -> str:
    lines = ["## NLP Analysis\n"]
    lines.append("### Summary")
    lines.append(nlp.get("summary", "No summary available."))
    lines.append("\n### Named Entities")
    entities = nlp.get("entities", [])
    if entities:
        for ent in entities[:10]:
            lines.append(
                f"- **{ent.get('text', '')}** "
                f"[{ent.get('label', '')}] "
                f"(confidence: {ent.get('score', 0):.2f})"
            )
    else:
        lines.append("No entities detected.")
    lines.append("\n### Topics")
    for t in nlp.get("topics", [])[:5]:
        bar = "█" * int(t["score"] * 30)
        lines.append(f"`{t['label']:<20}` {bar} {t['score']:.3f}")
    return "\n".join(lines)


def _format_vision(vision: dict | None) -> str:
    if vision is None:
        return "## Vision Analysis\n\nUpload a video file to enable vision analysis."
    lines = ["## Vision Analysis\n"]
    lines.append(f"**Frames processed:** {vision.get('frame_count', 0)}")
    lines.append(
        f"**Top visual label:** {vision.get('top_visual_label', 'unknown').title()}\n"
    )
    lines.append("### Frame Captions")
    for cap in vision.get("captions", [])[:8]:
        ts = cap.get("timestamp", 0)
        mm, ss = divmod(int(ts), 60)
        lines.append(f"**[{mm:02d}:{ss:02d}]** {cap.get('caption', '')}")
    lines.append("\n### Detected Objects")
    objs = vision.get("unique_objects", [])
    if objs:
        lines.append(", ".join(f"`{o}`" for o in objs))
    else:
        lines.append("No objects detected above threshold.")
    lines.append("\n### CLIP Scene Classification")
    for c in vision.get("clip_labels", [])[:5]:
        bar = "█" * int(c["score"] * 30)
        lines.append(f"`{c['label']:<30}` {bar} {c['score']:.3f}")
    return "\n".join(lines)


def _format_search_ready(stats: dict) -> str:
    lines = ["## Semantic Search Index\n"]
    lines.append(f"**Status:** {stats.get('status', 'unknown').title()}")
    lines.append(f"**Documents indexed:** {stats.get('document_count', 0)}")
    lines.append(f"**Embedding dimensions:** {stats.get('embedding_dim', 384)}")
    lines.append("\n**Sources indexed:**")
    for src in stats.get("sources", []):
        lines.append(f"- {src}")
    lines.append("\n*Use the search bar below to query across all sources.*")
    return "\n".join(lines)


# ── Gradio UI ──────────────────────────────────────────────────────────────────


def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="OmniSense — Multimodal AI Media Analyzer",
        theme=gr.themes.Soft(),
        css="""
        .header { text-align: center; padding: 1rem 0; }
        .header h1 { font-size: 2rem; font-weight: 700; }
        .header p  { color: #666; font-size: 1rem; }
        """,
    ) as demo:
        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML(
            """
        <div class="header">
            <h1>🎬 OmniSense</h1>
            <p>Multimodal AI Media Analyzer — Transcription · NLP · Vision · Semantic Search</p>
        </div>
        """
        )

        # ── Upload + Analyse ──────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                media_input = gr.File(
                    label="Upload Media File",
                    file_types=["video", "audio"],
                    type="filepath",
                )
                analyse_btn = gr.Button(
                    "🚀 Analyse",
                    variant="primary",
                    size="lg",
                )
            with gr.Column(scale=3):
                overview_out = gr.Markdown(
                    value="Upload a file and click Analyse to begin.",
                    label="Overview",
                )

        # ── Results tabs ──────────────────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("📝 Transcript"):
                transcript_out = gr.Markdown()
            with gr.Tab("🧠 NLP Analysis"):
                nlp_out = gr.Markdown()
            with gr.Tab("🖼 Vision Analysis"):
                vision_out = gr.Markdown()
            with gr.Tab("🔍 Search Index"):
                search_index_out = gr.Markdown()

        # ── Semantic search bar ───────────────────────────────────────────────
        gr.Markdown("---\n## 🔍 Semantic Search")
        gr.Markdown(
            "Search across transcript, summary, entities, and captions using natural language."
        )

        with gr.Row():
            search_input = gr.Textbox(
                placeholder="e.g. what did the speaker say about technology?",
                label="Search Query",
                scale=4,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top K",
                scale=1,
            )

        search_btn = gr.Button("Search", variant="secondary", interactive=False)
        search_out = gr.Markdown()

        # ── Example queries ───────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["what did the speaker talk about?"],
                ["what objects were detected in the video?"],
                ["who are the people mentioned?"],
                ["describe the visual scenes"],
                ["what is the main topic?"],
            ],
            inputs=search_input,
        )

        # ── Event handlers ────────────────────────────────────────────────────
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
            show_progress=True,
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
