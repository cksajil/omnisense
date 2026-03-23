"""
NLP analysis pipeline.

Runs three models over transcript chunks produced by AudioPipeline:
  1. facebook/bart-large-cnn      — extractive summarisation
  2. dslim/bert-base-NER          — named entity recognition
  3. facebook/bart-large-mnli     — zero-shot topic classification

Designed to receive the output of AudioPipeline.run() directly.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from transformers import (
    pipeline as hf_pipeline,
)

from omnisense.config import DEVICE, MODELS
from omnisense.pipelines.base import BasePipeline
from omnisense.utils.logger import log
from omnisense.utils.text import (
    aggregate_summaries,
    clean_text,
    merge_ner_entities,
    top_entities,
)

DEFAULT_TOPICS = [
    "technology",
    "politics",
    "sports",
    "science",
    "business",
    "health",
    "entertainment",
    "education",
    "environment",
    "travel",
]


class NLPPipeline(BasePipeline):
    """
    Runs summarisation, NER, and zero-shot classification on transcript text.

    Usage:
        audio_result = audio_pipeline("file.mp4")
        nlp = NLPPipeline()
        result = nlp(audio_result)

    Result shape:
        {
            "summary":         str,
            "chunk_summaries": list[str],
            "entities":        list[dict],
            "topics":          list[dict],
            "top_topic":       str,
            "word_count":      int,
            "models":          dict,
        }
    """

    def __init__(self, device: str = DEVICE) -> None:
        super().__init__(device=device)
        self._summarizer_model = None
        self._summarizer_tokenizer = None
        self._ner = None
        self._classifier = None
        self._hf_device = 0 if device == "cuda" else -1
        self._torch_device = torch.device("cuda" if device == "cuda" else "cpu")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load all three models. Downloads weights on first run (~1.5 GB)."""
        log.info("Loading NLP models — this may take a few minutes on first run…")

        log.info(f"  [1/3] Loading summarizer: {MODELS['summarizer']}")
        self._summarizer_tokenizer = AutoTokenizer.from_pretrained(MODELS["summarizer"])
        self._summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODELS["summarizer"]
        ).to(self._torch_device)
        self._summarizer_model.eval()

        log.info(f"  [2/3] Loading NER model: {MODELS['ner']}")
        self._ner = hf_pipeline(
            "token-classification",
            model=MODELS["ner"],
            aggregation_strategy="simple",
            device=self._hf_device,
        )

        log.info(f"  [3/3] Loading zero-shot classifier: {MODELS['classifier']}")
        self._classifier = hf_pipeline(
            "zero-shot-classification",
            model=MODELS["classifier"],
            device=self._hf_device,
        )

        log.info("All NLP models loaded ✓")

    # ── Core run ──────────────────────────────────────────────────────────────

    def run(
        self,
        audio_result: dict[str, Any],
        topics: list[str] | None = None,
        max_summary_length: int = 150,
        min_summary_length: int = 40,
    ) -> dict[str, Any]:
        """
        Run the full NLP analysis on transcript data.

        Args:
            audio_result: Output dict from AudioPipeline.run().
            topics: Custom topic labels for zero-shot classification.
            max_summary_length: Max tokens per chunk summary.
            min_summary_length: Min tokens per chunk summary.

        Returns:
            Structured NLP result dict.
        """
        self._validate_input(audio_result)

        transcript: str = clean_text(audio_result["transcript"])
        chunks: list[dict] = audio_result.get("chunks", [])
        topics = topics or DEFAULT_TOPICS

        if not transcript.strip():
            log.warning("Empty transcript received — returning empty NLP result")
            return self._empty_result()

        text_chunks = [c["text"] for c in chunks] if chunks else [transcript]

        log.info(f"Running NLP on {len(text_chunks)} chunk(s)…")

        # 1. Summarisation
        chunk_summaries = self._summarise_chunks(
            text_chunks, max_summary_length, min_summary_length
        )
        summary = aggregate_summaries(chunk_summaries)
        log.info(f"Summarisation complete — {len(chunk_summaries)} chunk summaries")

        # 2. NER
        raw_entities = self._extract_entities(transcript)
        entities = top_entities(merge_ner_entities(raw_entities))
        log.info(f"NER complete — {len(entities)} unique entities found")

        # 3. Zero-shot classification
        topic_result = self._classify_topics(summary or transcript[:512], topics)
        log.info(f"Classification complete — top topic: {topic_result[0]['label']}")

        return {
            "summary": summary,
            "chunk_summaries": chunk_summaries,
            "entities": entities,
            "topics": topic_result,
            "top_topic": topic_result[0]["label"] if topic_result else "unknown",
            "word_count": len(transcript.split()),
            "models": {
                "summarizer": MODELS["summarizer"],
                "ner": MODELS["ner"],
                "classifier": MODELS["classifier"],
            },
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _summarise_chunks(
        self,
        chunks: list[str],
        max_length: int,
        min_length: int,
    ) -> list[str]:
        """Summarise each chunk using BART directly via AutoModel."""
        summaries = []
        for i, chunk in enumerate(chunks):
            word_count = len(chunk.split())
            if word_count < 30:
                log.debug(f"Chunk {i} too short ({word_count} words) — using as-is")
                summaries.append(chunk.strip())
                continue
            try:
                inputs = self._summarizer_tokenizer(
                    chunk,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                ).to(self._torch_device)

                with torch.no_grad():
                    output_ids = self._summarizer_model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        min_new_tokens=min(min_length, word_count // 2),
                        num_beams=4,
                        early_stopping=True,
                    )

                summary = self._summarizer_tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                summaries.append(summary)

            except Exception as exc:
                log.warning(f"Summarisation failed for chunk {i}: {exc}")
                sentences = chunk.split(". ")[:2]
                summaries.append(". ".join(sentences))

        return summaries

    def _extract_entities(self, text: str) -> list[dict]:
        """Run token-classification NER over text."""
        max_chars = 512 * 4
        if len(text) <= max_chars:
            try:
                return self._ner(text)
            except Exception as exc:
                log.warning(f"NER failed: {exc}")
                return []

        all_entities: list[dict] = []
        for start in range(0, len(text), max_chars):
            window = text[start : start + max_chars]
            try:
                entities = self._ner(window)
                for ent in entities:
                    ent["start"] = ent.get("start", 0) + start
                    ent["end"] = ent.get("end", 0) + start
                all_entities.extend(entities)
            except Exception as exc:
                log.warning(f"NER failed on window at {start}: {exc}")

        return all_entities

    def _classify_topics(self, text: str, topics: list[str]) -> list[dict]:
        """Run zero-shot classification."""
        try:
            result = self._classifier(
                text[:1024],
                candidate_labels=topics,
                multi_label=False,
            )
            return [
                {"label": label, "score": round(score, 4)}
                for label, score in zip(result["labels"], result["scores"])
            ]
        except Exception as exc:
            log.warning(f"Zero-shot classification failed: {exc}")
            return [{"label": "unknown", "score": 0.0}]

    def _validate_input(self, audio_result: dict) -> None:
        if not isinstance(audio_result, dict):
            raise ValueError(
                f"audio_result must be a dict, got {type(audio_result).__name__}"
            )
        if "transcript" not in audio_result:
            raise ValueError(
                "audio_result must contain 'transcript' key. "
                "Pass the direct output of AudioPipeline.run()."
            )

    def _empty_result(self) -> dict:
        return {
            "summary": "",
            "chunk_summaries": [],
            "entities": [],
            "topics": [{"label": "unknown", "score": 0.0}],
            "top_topic": "unknown",
            "word_count": 0,
            "models": {
                "summarizer": MODELS["summarizer"],
                "ner": MODELS["ner"],
                "classifier": MODELS["classifier"],
            },
        }
