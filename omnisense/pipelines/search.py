"""
omnisense/pipelines/search.py

Semantic search over timestamped transcript chunks.

Stack:
  - sentence-transformers/all-MiniLM-L6-v2: 80MB, fast on CPU, great quality
  - FAISS IndexFlatIP: exact cosine search (inner product on normalized vectors)
    No approximation needed — transcript chunk counts are small (< 10k segments)

Design notes:
  - TranscriptSearchIndex is stateful: build() once, search() many times.
  - Cosine similarity via normalized vectors + inner product avoids the need
    for IndexFlatL2 + manual normalization.
  - min_score acts as a quality gate: below 0.30 is typically noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from omnisense.pipelines.audio import TranscriptChunk

# ── Constants ──────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 384-dim embeddings, 80MB model, ~14k sentences/sec on CPU


# ── Data types ─────────────────────────────────────────────────────────────────


@dataclass
class SearchHit:
    """A single search result linking a query to a transcript segment."""

    chunk: TranscriptChunk
    score: float  # cosine similarity in [0, 1]; higher = more relevant
    rank: int  # 1-based position in result list


# ── Index ──────────────────────────────────────────────────────────────────────


class TranscriptSearchIndex:
    """
    FAISS-backed semantic index over TranscriptChunk objects.

    Usage:
        index = TranscriptSearchIndex()
        index.build(chunks)           # call once after transcription
        hits = index.search("query")  # call as many times as needed
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[TranscriptChunk] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, chunks: list[TranscriptChunk]) -> None:
        """
        Encode all chunks and build the FAISS index.

        This is called once after transcription completes.
        Subsequent calls replace the existing index.

        Args:
            chunks: List of TranscriptChunk from the audio pipeline.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        self._chunks = chunks
        texts = [c.text for c in chunks]

        logger.info(f"Encoding {len(texts)} segments...")
        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize -> cosine via inner product
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)  # exact inner product search
        self._index.add(embeddings)

        logger.info(f"Index built: {self._index.ntotal} vectors, dim={dim}")

    # ── Search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.30,
    ) -> list[SearchHit]:
        """
        Find transcript segments semantically matching the query.

        Args:
            query:      Natural language search string.
            top_k:      Maximum number of results to return.
            min_score:  Minimum cosine similarity threshold [0, 1].
                        0.30 is a generous default that captures paraphrases.
                        Raise to 0.50+ for stricter, more precise results.

        Returns:
            List of SearchHit sorted by score descending.
            Empty list if no segments meet the min_score threshold.

        Raises:
            RuntimeError: If build() has not been called yet.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() before search().")

        if not query.strip():
            return []

        # Encode and normalize query
        q_embedding: np.ndarray = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # FAISS search returns (scores, indices) arrays of shape (1, top_k)
        scores, indices = self._index.search(
            q_embedding, min(top_k, self._index.ntotal)
        )

        hits: list[SearchHit] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx == -1:
                continue  # FAISS padding for under-filled results
            similarity = float(score)
            if similarity < min_score:
                continue
            hits.append(
                SearchHit(
                    chunk=self._chunks[idx],
                    score=round(similarity, 4),
                    rank=rank,
                )
            )

        logger.info(
            f"Search '{query[:60]}' -> {len(hits)} hits "
            f"(min_score={min_score}, top_k={top_k})"
        )
        return hits

    # ── Introspection ──────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True if the index has been built and is ready to search."""
        return self._index is not None and self._index.ntotal > 0

    @property
    def segment_count(self) -> int:
        """Number of indexed segments."""
        return len(self._chunks)
