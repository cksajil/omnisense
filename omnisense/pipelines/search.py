"""
omnisense/pipelines/search.py

Semantic search over timestamped transcript chunks.
Uses sentence-transformers (MiniLM) + FAISS for CPU-friendly semantic search.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from omnisense.pipelines.audio import TranscriptChunk

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SearchHit:
    chunk: TranscriptChunk
    score: float
    rank: int


class TranscriptSearchIndex:

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._index = None
        self._chunks: list[TranscriptChunk] = []

    def build(self, chunks: list[TranscriptChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")
        self._chunks = chunks
        texts = [c.text for c in chunks]
        logger.info(f"Encoding {len(texts)} segments...")
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)
        logger.info(f"Index built: {self._index.ntotal} vectors, dim={dim}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
    ) -> list[SearchHit]:
        if self._index is None:
            raise RuntimeError("Index not built. Call build() before search().")
        query = query.strip()
        if not query:
            return []
        q_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, indices = self._index.search(
            q_embedding, min(top_k, self._index.ntotal)
        )
        hits: list[SearchHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue
            similarity = float(score)
            if similarity < min_score:
                continue
            hits.append(
                SearchHit(
                    chunk=self._chunks[int(idx)],
                    score=round(similarity, 4),
                    rank=rank,
                )
            )
        logger.info(f"Search '{query[:60]}' -> {len(hits)} hits")
        return hits

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def segment_count(self) -> int:
        return len(self._chunks)
