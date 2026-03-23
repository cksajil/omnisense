"""Semantic search pipeline — Sentence Transformers + FAISS."""

from omnisense.pipelines.base import BasePipeline


class SearchPipeline(BasePipeline):
    """Phase 4 implementation."""

    def load(self) -> None:  # noqa: D102
        pass

    def run(self, query: str) -> dict:  # noqa: D102
        return {}
