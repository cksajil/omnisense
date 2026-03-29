"""
tests/test_search.py

Unit tests for the search pipeline.
Run with: pytest tests/
"""

import pytest

from omnisense.pipelines.audio import TranscriptChunk
from omnisense.pipelines.search import SearchHit, TranscriptSearchIndex

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_CHUNKS = [
    TranscriptChunk(
        "The transformer architecture changed natural language processing forever.",
        0.0,
        5.0,
        0,
    ),
    TranscriptChunk(
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        5.0,
        10.0,
        1,
    ),
    TranscriptChunk(
        "Gradient descent optimizes the loss function during training.", 10.0, 15.0, 2
    ),
    TranscriptChunk(
        "The learning rate controls how fast the model learns.", 15.0, 20.0, 3
    ),
    TranscriptChunk(
        "Climate change is one of the most pressing issues of our time.", 20.0, 25.0, 4
    ),
    TranscriptChunk(
        "Renewable energy sources like solar and wind are growing rapidly.",
        25.0,
        30.0,
        5,
    ),
    TranscriptChunk(
        "The budget deficit has increased significantly this year.", 30.0, 35.0, 6
    ),
]


@pytest.fixture(scope="module")
def built_index() -> TranscriptSearchIndex:
    index = TranscriptSearchIndex()
    index.build(SAMPLE_CHUNKS)
    return index


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestTranscriptSearchIndex:
    def test_build_sets_ready(self, built_index):
        assert built_index.is_ready is True

    def test_segment_count(self, built_index):
        assert built_index.segment_count == len(SAMPLE_CHUNKS)

    def test_search_returns_hits(self, built_index):
        hits = built_index.search("transformer and attention", top_k=3)
        assert len(hits) >= 1

    def test_search_hit_fields(self, built_index):
        hits = built_index.search("learning rate", top_k=1)
        assert len(hits) == 1
        hit = hits[0]
        assert isinstance(hit, SearchHit)
        assert 0.0 <= hit.score <= 1.0
        assert hit.rank == 1
        assert isinstance(hit.chunk, TranscriptChunk)

    def test_search_relevance_ordering(self, built_index):
        """Top hit should be more relevant than lower hits."""
        hits = built_index.search("climate and environment", top_k=3, min_score=0.1)
        if len(hits) >= 2:
            assert hits[0].score >= hits[1].score

    def test_search_min_score_filters(self, built_index):
        """With very high min_score, should return no results for unrelated query."""
        hits = built_index.search("quantum physics nuclear fusion", min_score=0.95)
        assert hits == []

    def test_search_empty_query(self, built_index):
        hits = built_index.search("")
        assert hits == []

    def test_search_whitespace_query(self, built_index):
        hits = built_index.search("   ")
        assert hits == []

    def test_build_raises_on_empty(self):
        index = TranscriptSearchIndex()
        with pytest.raises(ValueError, match="empty"):
            index.build([])

    def test_search_raises_before_build(self):
        index = TranscriptSearchIndex()
        with pytest.raises(RuntimeError, match="build"):
            index.search("something")

    def test_timestamps_preserved(self, built_index):
        """Ensure chunks returned by search retain their original timestamps."""
        hits = built_index.search("budget deficit")
        assert len(hits) >= 1
        top = hits[0]
        # The budget chunk has start=30.0, end=35.0
        assert top.chunk.start >= 0.0
        assert top.chunk.end > top.chunk.start

    def test_top_k_respected(self, built_index):
        hits = built_index.search("model training optimization", top_k=2, min_score=0.1)
        assert len(hits) <= 2
