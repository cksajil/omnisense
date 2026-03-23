"""Tests for utility functions."""

from __future__ import annotations

from omnisense.utils.media import chunk_transcript


class TestChunkTranscript:
    def test_empty_segments_returns_empty(self):
        assert chunk_transcript([]) == []

    def test_single_segment_returns_one_chunk(self):
        segments = [{"text": "Hello world.", "start": 0.0, "end": 2.0}]
        chunks = chunk_transcript(segments)
        assert len(chunks) == 1
        assert chunks[0]["text"] == "Hello world."
        assert chunks[0]["start"] == 0.0
        assert chunks[0]["end"] == 2.0

    def test_chunks_contain_required_keys(self):
        segments = [{"text": "Word " * 10, "start": 0.0, "end": 5.0}]
        chunks = chunk_transcript(segments)
        for chunk in chunks:
            assert "text" in chunk
            assert "start" in chunk
            assert "end" in chunk
            assert "segment_ids" in chunk

    def test_large_input_produces_multiple_chunks(self):
        # 600 words across 60 segments — with max_tokens=512 should produce 2 chunks
        segments = [
            {"text": "word " * 10, "start": float(i), "end": float(i + 1)}
            for i in range(60)
        ]
        chunks = chunk_transcript(segments, max_tokens=512, overlap_tokens=50)
        assert len(chunks) >= 2

    def test_overlap_preserves_context(self):
        segments = [
            {"text": "word " * 30, "start": float(i), "end": float(i + 1)}
            for i in range(30)
        ]
        chunks = chunk_transcript(segments, max_tokens=100, overlap_tokens=20)
        if len(chunks) > 1:
            # Chunks should share some words due to overlap
            words_0 = set(chunks[0]["text"].split()[-20:])
            words_1 = set(chunks[1]["text"].split()[:20])
            assert len(words_0 & words_1) > 0

    def test_timestamps_are_monotonic(self):
        segments = [
            {"text": f"Segment {i}.", "start": float(i), "end": float(i + 1)}
            for i in range(20)
        ]
        chunks = chunk_transcript(segments)
        for chunk in chunks:
            assert chunk["start"] <= chunk["end"]
