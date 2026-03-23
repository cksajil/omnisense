"""Tests for the semantic search pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from omnisense.pipelines.search import SearchPipeline


def make_mock_encoder():
    """Return a mock SentenceTransformer that returns deterministic embeddings."""
    encoder = MagicMock()
    encoder.encode = MagicMock(
        side_effect=lambda texts, **kwargs: np.random.rand(
            len(texts) if isinstance(texts, list) else 1, 384
        ).astype(np.float32)
    )
    return encoder


def make_loaded_pipeline() -> SearchPipeline:
    pipeline = SearchPipeline(device="cpu")
    pipeline._loaded = True
    pipeline._encoder = make_mock_encoder()
    return pipeline


def make_audio_result() -> dict:
    return {
        "transcript": "This is a test about artificial intelligence.",
        "segments": [
            {"text": "This is a test", "start": 0.0, "end": 2.0},
            {"text": "about artificial intelligence.", "start": 2.0, "end": 4.0},
        ],
        "chunks": [
            {
                "text": "This is a test about artificial intelligence.",
                "start": 0.0,
                "end": 4.0,
            },
        ],
    }


def make_nlp_result() -> dict:
    return {
        "summary": "A test about artificial intelligence.",
        "chunk_summaries": ["AI test summary."],
        "entities": [
            {"text": "artificial intelligence", "label": "MISC", "score": 0.95},
        ],
        "top_topic": "technology",
    }


def make_vision_result() -> dict:
    return {
        "captions": [
            {"frame_id": 0, "timestamp": 0.0, "caption": "a person at a computer"},
            {"frame_id": 1, "timestamp": 1.0, "caption": "a screen with code"},
        ],
        "unique_objects": ["person", "laptop", "chair"],
    }


class TestSearchPipeline:
    def test_query_raises_if_index_not_built(self):
        pipeline = make_loaded_pipeline()
        with pytest.raises(RuntimeError, match="Index not built"):
            pipeline.query("test query")

    def test_build_index_returns_stats(self):
        pipeline = make_loaded_pipeline()
        stats = pipeline.build_index(
            audio_result=make_audio_result(),
            nlp_result=make_nlp_result(),
        )
        assert "document_count" in stats
        assert stats["document_count"] > 0
        assert "embedding_dim" in stats
        assert "sources" in stats

    def test_query_returns_list(self):
        pipeline = make_loaded_pipeline()
        pipeline.build_index(
            audio_result=make_audio_result(),
            nlp_result=make_nlp_result(),
        )
        results = pipeline.query("artificial intelligence")
        assert isinstance(results, list)

    def test_query_result_has_expected_keys(self):
        pipeline = make_loaded_pipeline()
        pipeline.build_index(audio_result=make_audio_result())
        results = pipeline.query("test", top_k=1)
        assert len(results) > 0
        keys = set(results[0].keys())
        assert keys == {"text", "source", "score", "metadata"}

    def test_top_k_respected(self):
        pipeline = make_loaded_pipeline()
        pipeline.build_index(
            audio_result=make_audio_result(),
            nlp_result=make_nlp_result(),
            vision_result=make_vision_result(),
        )
        results = pipeline.query("technology", top_k=2)
        assert len(results) <= 2

    def test_empty_query_returns_empty(self):
        pipeline = make_loaded_pipeline()
        pipeline.build_index(audio_result=make_audio_result())
        results = pipeline.query("")
        assert results == []

    def test_all_three_sources_indexed(self):
        pipeline = make_loaded_pipeline()
        stats = pipeline.build_index(
            audio_result=make_audio_result(),
            nlp_result=make_nlp_result(),
            vision_result=make_vision_result(),
        )
        assert "transcript" in stats["sources"]
        assert "summary" in stats["sources"]
        assert "caption" in stats["sources"]

    def test_empty_input_returns_zero_docs(self):
        pipeline = make_loaded_pipeline()
        stats = pipeline.build_index()
        assert stats["document_count"] == 0

    def test_save_and_load_index(self, tmp_path):
        pipeline = make_loaded_pipeline()
        pipeline.build_index(audio_result=make_audio_result())
        pipeline.save_index(tmp_path)

        assert (tmp_path / "index.faiss").exists()
        assert (tmp_path / "documents.json").exists()

        new_pipeline = make_loaded_pipeline()
        new_pipeline.load_index(tmp_path)
        assert new_pipeline._index_built
        assert len(new_pipeline._documents) > 0

    def test_get_stats_before_build(self):
        pipeline = make_loaded_pipeline()
        stats = pipeline.get_stats()
        assert stats["status"] == "not built"

    def test_get_stats_after_build(self):
        pipeline = make_loaded_pipeline()
        pipeline.build_index(audio_result=make_audio_result())
        stats = pipeline.get_stats()
        assert stats["status"] == "ready"
        assert stats["document_count"] > 0
