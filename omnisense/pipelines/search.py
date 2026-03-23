"""
Semantic search pipeline.

Encodes all OmniSense pipeline outputs into a unified FAISS vector store,
enabling natural language search over transcripts, captions, summaries,
and named entities from any analysed media file.

Architecture:
    1. Collect text chunks from audio, nlp, and vision results
    2. Encode with Sentence Transformers (all-MiniLM-L6-v2)
    3. Index with FAISS (flat L2 — exact search, no approximation)
    4. Query with natural language → top-k most similar chunks
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from omnisense.config import CACHE_DIR, DEVICE, MODELS
from omnisense.pipelines.base import BasePipeline
from omnisense.utils.logger import log


class SearchPipeline(BasePipeline):
    """
    Builds and queries a FAISS semantic search index over media analysis results.

    Usage:
        # Build index from pipeline results
        search = SearchPipeline()
        search.build_index(audio_result, nlp_result, vision_result)

        # Query in natural language
        results = search.query("who spoke about technology?", top_k=5)

    Result shape from query():
        [
            {
                "text":       str,    # matched text chunk
                "source":     str,    # "transcript" | "summary" | "caption" | "entity"
                "score":      float,  # cosine similarity score (higher = more similar)
                "metadata":   dict,   # timestamp, frame_id, etc.
            },
            ...
        ]
    """

    def __init__(self, device: str = DEVICE) -> None:
        super().__init__(device=device)
        self._encoder: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._documents: list[dict] = []
        self._index_built = False
        # SentenceTransformers uses "cuda" or "cpu" directly
        self._st_device = "cuda" if device == "cuda" else "cpu"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load the sentence transformer encoder."""
        log.info(f"Loading encoder: {MODELS['embedder']}")
        self._encoder = SentenceTransformer(
            MODELS["embedder"],
            device=self._st_device,
            cache_folder=str(CACHE_DIR / "embeddings"),
        )
        log.info("Encoder loaded ✓")

    def run(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Alias for query() — satisfies BasePipeline interface.
        Requires build_index() to be called first.
        """
        return self.query(query, top_k=top_k)

    # ── Index building ────────────────────────────────────────────────────────

    def build_index(
        self,
        audio_result: dict[str, Any] | None = None,
        nlp_result: dict[str, Any] | None = None,
        vision_result: dict[str, Any] | None = None,
        extra_documents: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        Build a FAISS index from pipeline outputs.

        Each pipeline result is decomposed into individual text chunks,
        each tagged with its source and metadata. All chunks are encoded
        and added to the FAISS index in one batch.

        Args:
            audio_result:    Output of AudioPipeline.run().
            nlp_result:      Output of NLPPipeline.run().
            vision_result:   Output of VisionPipeline.run().
            extra_documents: Additional custom dicts with 'text' and 'source'.

        Returns:
            Index stats dict: {document_count, embedding_dim, sources}
        """
        if not self._loaded:
            self.load()
            self._loaded = True

        documents: list[dict] = []

        # ── Audio — transcript segments ───────────────────────────────────────
        if audio_result:
            for seg in audio_result.get("segments", []):
                text = seg.get("text", "").strip()
                if text:
                    documents.append(
                        {
                            "text": text,
                            "source": "transcript",
                            "metadata": {
                                "start": seg.get("start", 0),
                                "end": seg.get("end", 0),
                            },
                        }
                    )

            # Also index the full transcript chunks for broader context
            for chunk in audio_result.get("chunks", []):
                text = chunk.get("text", "").strip()
                if text:
                    documents.append(
                        {
                            "text": text,
                            "source": "transcript_chunk",
                            "metadata": {
                                "start": chunk.get("start", 0),
                                "end": chunk.get("end", 0),
                            },
                        }
                    )

        # ── NLP — summary + entities ──────────────────────────────────────────
        if nlp_result:
            summary = nlp_result.get("summary", "").strip()
            if summary:
                documents.append(
                    {
                        "text": summary,
                        "source": "summary",
                        "metadata": {"top_topic": nlp_result.get("top_topic", "")},
                    }
                )

            for chunk_summary in nlp_result.get("chunk_summaries", []):
                if chunk_summary.strip():
                    documents.append(
                        {
                            "text": chunk_summary.strip(),
                            "source": "chunk_summary",
                            "metadata": {},
                        }
                    )

            for entity in nlp_result.get("entities", []):
                text = entity.get("text", "").strip()
                if text:
                    documents.append(
                        {
                            "text": f"{entity.get('label', '')}: {text}",
                            "source": "entity",
                            "metadata": {
                                "label": entity.get("label", ""),
                                "score": entity.get("score", 0.0),
                            },
                        }
                    )

        # ── Vision — captions + detected objects ──────────────────────────────
        if vision_result:
            for cap in vision_result.get("captions", []):
                text = cap.get("caption", "").strip()
                if text and text != "caption unavailable":
                    documents.append(
                        {
                            "text": text,
                            "source": "caption",
                            "metadata": {
                                "frame_id": cap.get("frame_id", 0),
                                "timestamp": cap.get("timestamp", 0),
                            },
                        }
                    )

            unique_objects = vision_result.get("unique_objects", [])
            if unique_objects:
                documents.append(
                    {
                        "text": "Objects detected: " + ", ".join(unique_objects),
                        "source": "objects",
                        "metadata": {"count": len(unique_objects)},
                    }
                )

        # ── Extra documents ───────────────────────────────────────────────────
        if extra_documents:
            documents.extend(extra_documents)

        if not documents:
            log.warning(
                "No documents to index — build_index() called with empty results"
            )
            return {"document_count": 0, "embedding_dim": 0, "sources": []}

        log.info(f"Encoding {len(documents)} documents…")
        texts = [doc["text"] for doc in documents]
        embeddings = self._encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalise for cosine similarity
        )

        # FAISS IndexFlatIP = inner product on normalised vectors = cosine similarity
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype(np.float32))
        self._documents = documents
        self._index_built = True

        sources = list(set(doc["source"] for doc in documents))
        log.info(
            f"Index built — {len(documents)} docs, " f"dim={dim}, sources={sources}"
        )

        return {
            "document_count": len(documents),
            "embedding_dim": dim,
            "sources": sources,
        }

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        """
        Search the index with a natural language query.

        Args:
            query_text: Natural language search query.
            top_k: Number of results to return.

        Returns:
            List of result dicts sorted by similarity score descending.

        Raises:
            RuntimeError: If build_index() has not been called yet.
        """
        if not self._index_built:
            raise RuntimeError(
                "Index not built. Call build_index() with pipeline results first."
            )

        if not query_text.strip():
            return []

        # Encode and normalise the query
        query_embedding = self._encoder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        # Search — returns distances and indices
        k = min(top_k, len(self._documents))
        scores, indices = self._index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self._documents[idx]
            results.append(
                {
                    "text": doc["text"],
                    "source": doc["source"],
                    "score": round(float(score), 4),
                    "metadata": doc.get("metadata", {}),
                }
            )

        log.debug(f"Query '{query_text[:50]}' → {len(results)} results")
        return results

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_index(self, output_dir: str | Path) -> Path:
        """
        Persist the FAISS index and document store to disk.

        Useful for caching results between sessions so you don't
        re-encode on every run.

        Args:
            output_dir: Directory to write index files.

        Returns:
            Path to the saved index directory.
        """
        if not self._index_built:
            raise RuntimeError("No index to save. Call build_index() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(output_dir / "index.faiss"))

        with open(output_dir / "documents.json", "w") as f:
            # Strip PIL images from metadata before serialising
            safe_docs = [
                {k: v for k, v in doc.items() if k != "image"}
                for doc in self._documents
            ]
            json.dump(safe_docs, f, indent=2)

        log.info(f"Index saved to {output_dir}")
        return output_dir

    def load_index(self, index_dir: str | Path) -> None:
        """
        Load a previously saved FAISS index from disk.

        Args:
            index_dir: Directory containing index.faiss and documents.json.
        """
        if not self._loaded:
            self.load()
            self._loaded = True

        index_dir = Path(index_dir)
        index_path = index_dir / "index.faiss"
        docs_path = index_dir / "documents.json"

        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                f"Index files not found in {index_dir}. "
                "Run build_index() and save_index() first."
            )

        self._index = faiss.read_index(str(index_path))

        with open(docs_path) as f:
            self._documents = json.load(f)

        self._index_built = True
        log.info(
            f"Index loaded from {index_dir} " f"— {len(self._documents)} documents"
        )

    def get_stats(self) -> dict:
        """Return current index statistics."""
        if not self._index_built:
            return {"status": "not built"}
        return {
            "status": "ready",
            "document_count": len(self._documents),
            "index_size": self._index.ntotal,
            "sources": list(set(d["source"] for d in self._documents)),
        }
