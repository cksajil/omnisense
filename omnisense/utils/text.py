"""
Text processing utilities for the NLP pipeline.
Handles cleaning, merging, and post-processing of model outputs.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """
    Normalise raw transcript text before passing to NLP models.

    - Collapses multiple spaces/newlines
    - Strips leading/trailing whitespace
    - Removes filler words common in ASR output

    Args:
        text: Raw transcript string.

    Returns:
        Cleaned string.
    """
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove common ASR artifacts
    text = re.sub(r"\b(um|uh|hmm|mhm|uh-huh)\b", "", text, flags=re.IGNORECASE)
    # Clean up any double spaces left behind
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def merge_ner_entities(raw_entities: list[dict]) -> list[dict]:
    """
    Merge consecutive subword/word tokens from HuggingFace NER output
    into clean, readable entity spans.

    HuggingFace NER returns one dict per token — e.g. "New", "York" as
    separate B-LOC / I-LOC tokens. This function joins them into
    {"text": "New York", "label": "LOC", "score": 0.97}.

    Args:
        raw_entities: List of raw HF NER dicts with keys:
                      word, entity_group, score, start, end.

    Returns:
        Deduplicated list of merged entity dicts.
    """
    if not raw_entities:
        return []

    merged: list[dict] = []
    current: dict | None = None

    for ent in raw_entities:
        word = ent.get("word", "").replace("##", "")  # strip BERT subword prefix
        group = ent.get("entity_group", ent.get("entity", ""))
        score = ent.get("score", 0.0)
        start = ent.get("start", 0)
        end = ent.get("end", 0)

        # Strip B- / I- prefixes if aggregation wasn't done upstream
        label = re.sub(r"^[BI]-", "", group)

        if current is None:
            current = {
                "text": word,
                "label": label,
                "score": score,
                "start": start,
                "end": end,
            }
        elif label == current["label"] and start <= current["end"] + 2:
            # Continuation — extend current entity
            current["text"] = f"{current['text']} {word}".strip()
            current["end"] = end
            current["score"] = (current["score"] + score) / 2
        else:
            merged.append(current)
            current = {
                "text": word,
                "label": label,
                "score": score,
                "start": start,
                "end": end,
            }

    if current:
        merged.append(current)

    # Deduplicate by (text, label) keeping highest score
    seen: dict[tuple, dict] = {}
    for ent in merged:
        key = (ent["text"].lower(), ent["label"])
        if key not in seen or ent["score"] > seen[key]["score"]:
            seen[key] = ent

    return sorted(seen.values(), key=lambda e: e["score"], reverse=True)


def aggregate_summaries(summaries: list[str]) -> str:
    """
    Combine per-chunk summaries into a single coherent summary.

    For short inputs (1–2 chunks) just joins with a space.
    For longer inputs, deduplicates repeated sentences that
    BART tends to produce across overlapping chunks.

    Args:
        summaries: List of summary strings, one per chunk.

    Returns:
        Single aggregated summary string.
    """
    if not summaries:
        return ""
    if len(summaries) == 1:
        return summaries[0]

    # Split into sentences, deduplicate while preserving order
    seen_sentences: set[str] = set()
    final_sentences: list[str] = []

    for summary in summaries:
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        for sent in sentences:
            normalised = sent.lower().strip()
            if normalised and normalised not in seen_sentences:
                seen_sentences.add(normalised)
                final_sentences.append(sent.strip())

    return " ".join(final_sentences)


def top_entities(entities: list[dict], top_n: int = 10) -> list[dict]:
    """
    Return the top N most confident entities, one per unique text span.

    Args:
        entities: Merged entity list from merge_ner_entities().
        top_n: Maximum number of entities to return.

    Returns:
        Top N entities sorted by score descending.
    """
    return entities[:top_n]
