"""
Tests for the NLP pipeline and text utilities.
All model calls are mocked — no weights downloaded during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from omnisense.pipelines.nlp import NLPPipeline
from omnisense.utils.text import (
    aggregate_summaries,
    clean_text,
    merge_ner_entities,
    top_entities,
)

# ── Text utilities ────────────────────────────────────────────────────────────


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"

    def test_strips_filler_words(self):
        result = clean_text("um so uh this is hmm a test")
        assert "um" not in result
        assert "uh" not in result
        assert "hmm" not in result

    def test_preserves_meaningful_content(self):
        text = "The quick brown fox jumps over the lazy dog."
        assert clean_text(text) == text

    def test_handles_empty_string(self):
        assert clean_text("") == ""


class TestMergeNerEntities:
    def test_empty_input(self):
        assert merge_ner_entities([]) == []

    def test_merges_consecutive_tokens(self):
        raw = [
            {"word": "New", "entity_group": "LOC", "score": 0.99, "start": 0, "end": 3},
            {
                "word": "York",
                "entity_group": "LOC",
                "score": 0.98,
                "start": 4,
                "end": 8,
            },
        ]
        merged = merge_ner_entities(raw)
        assert len(merged) == 1
        assert merged[0]["text"] == "New York"
        assert merged[0]["label"] == "LOC"

    def test_separates_different_entity_types(self):
        raw = [
            {
                "word": "Apple",
                "entity_group": "ORG",
                "score": 0.95,
                "start": 0,
                "end": 5,
            },
            {
                "word": "London",
                "entity_group": "LOC",
                "score": 0.97,
                "start": 10,
                "end": 16,
            },
        ]
        merged = merge_ner_entities(raw)
        assert len(merged) == 2

    def test_deduplicates_same_entity(self):
        raw = [
            {
                "word": "Apple",
                "entity_group": "ORG",
                "score": 0.95,
                "start": 0,
                "end": 5,
            },
            {
                "word": "Apple",
                "entity_group": "ORG",
                "score": 0.90,
                "start": 20,
                "end": 25,
            },
        ]
        merged = merge_ner_entities(raw)
        assert len(merged) == 1
        assert merged[0]["score"] == 0.95  # keeps highest score

    def test_strips_bi_prefixes(self):
        raw = [
            {
                "word": "Paris",
                "entity_group": "B-LOC",
                "score": 0.99,
                "start": 0,
                "end": 5,
            },
        ]
        merged = merge_ner_entities(raw)
        assert merged[0]["label"] == "LOC"


class TestAggregateSummaries:
    def test_empty_returns_empty_string(self):
        assert aggregate_summaries([]) == ""

    def test_single_summary_returned_as_is(self):
        assert aggregate_summaries(["Hello world."]) == "Hello world."

    def test_deduplicates_repeated_sentences(self):
        s = "The cat sat on the mat."
        result = aggregate_summaries([s, s])
        assert result.count("The cat sat on the mat.") == 1

    def test_joins_unique_sentences(self):
        result = aggregate_summaries(["First sentence.", "Second sentence."])
        assert "First sentence." in result
        assert "Second sentence." in result


class TestTopEntities:
    def test_limits_to_top_n(self):
        entities = [
            {"text": f"Entity{i}", "label": "ORG", "score": float(i) / 10}
            for i in range(20)
        ]
        assert len(top_entities(entities, top_n=5)) == 5

    def test_preserves_order(self):
        entities = [
            {"text": "A", "label": "ORG", "score": 0.9},
            {"text": "B", "label": "ORG", "score": 0.8},
        ]
        result = top_entities(entities)
        assert result[0]["text"] == "A"


# ── NLPPipeline ───────────────────────────────────────────────────────────────


def make_audio_result(
    transcript: str = "This is a test transcript about technology.",
) -> dict:
    """Helper — minimal valid AudioPipeline output."""
    return {
        "transcript": transcript,
        "chunks": [{"text": transcript, "start": 0.0, "end": 5.0, "segment_ids": [0]}],
        "language": "en",
        "duration": 5.0,
        "model": "openai/whisper-base",
    }


class TestNLPPipeline:
    def _make_loaded_pipeline(self) -> NLPPipeline:
        """Return a pipeline with mocked models — no weights downloaded."""
        pipeline = NLPPipeline(device="cpu")
        pipeline._loaded = True

        pipeline._summarizer = MagicMock(
            return_value=[{"summary_text": "A test summary about technology."}]
        )
        pipeline._ner = MagicMock(
            return_value=[
                {
                    "word": "OpenAI",
                    "entity_group": "ORG",
                    "score": 0.99,
                    "start": 0,
                    "end": 6,
                },
            ]
        )
        pipeline._classifier = MagicMock(
            return_value={
                "labels": ["technology", "science", "business"],
                "scores": [0.85, 0.10, 0.05],
            }
        )
        return pipeline

    def test_raises_on_missing_transcript_key(self):
        pipeline = self._make_loaded_pipeline()
        with pytest.raises(ValueError, match="transcript"):
            pipeline.run({"chunks": []})

    def test_raises_on_non_dict_input(self):
        pipeline = self._make_loaded_pipeline()
        with pytest.raises(ValueError):
            pipeline.run("raw string input")

    def test_result_has_expected_keys(self):
        pipeline = self._make_loaded_pipeline()
        result = pipeline.run(make_audio_result())
        expected = {
            "summary",
            "chunk_summaries",
            "entities",
            "topics",
            "top_topic",
            "word_count",
            "models",
        }
        assert set(result.keys()) == expected

    def test_empty_transcript_returns_empty_result(self):
        pipeline = self._make_loaded_pipeline()
        result = pipeline.run(make_audio_result(transcript=""))
        assert result["summary"] == ""
        assert result["entities"] == []
        assert result["top_topic"] == "unknown"

    def test_top_topic_matches_highest_score(self):
        pipeline = self._make_loaded_pipeline()
        result = pipeline.run(make_audio_result())
        assert result["top_topic"] == "technology"

    def test_word_count_is_accurate(self):
        transcript = "one two three four five"
        pipeline = self._make_loaded_pipeline()
        result = pipeline.run(make_audio_result(transcript=transcript))
        assert result["word_count"] == 5

    def test_models_dict_contains_all_three(self):
        pipeline = self._make_loaded_pipeline()
        result = pipeline.run(make_audio_result())
        assert "summarizer" in result["models"]
        assert "ner" in result["models"]
        assert "classifier" in result["models"]

    def test_custom_topics_are_used(self):
        pipeline = self._make_loaded_pipeline()
        custom_topics = ["finance", "crypto", "real estate"]
        pipeline._classifier = MagicMock(
            return_value={
                "labels": ["finance", "crypto", "real estate"],
                "scores": [0.7, 0.2, 0.1],
            }
        )
        result = pipeline.run(make_audio_result(), topics=custom_topics)
        assert result["top_topic"] == "finance"

    def test_graceful_degradation_on_summarizer_failure(self):
        pipeline = self._make_loaded_pipeline()
        pipeline._summarizer = MagicMock(side_effect=RuntimeError("model error"))
        # Should not raise — falls back to first 2 sentences
        result = pipeline.run(make_audio_result())
        assert isinstance(result["summary"], str)

    def test_no_chunks_falls_back_to_full_transcript(self):
        pipeline = self._make_loaded_pipeline()
        audio_result = make_audio_result()
        audio_result["chunks"] = []  # no chunks
        result = pipeline.run(audio_result)
        assert result["summary"] != ""
