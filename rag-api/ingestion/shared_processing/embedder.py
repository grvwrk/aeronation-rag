"""Embedder: Chunk objects -> EmbeddedChunk objects with dense + sparse embeddings.

Implements the Embedding stage for hybrid retrieval in the Aeronation-RAG data ingestion pipeline:
1. Dense embeddings: Batch generation using FastEmbed / ONNX models (e.g. all-MiniLM-L6-v2).
2. Sparse embeddings: Batch generation using FastEmbed sparse text models (e.g. Qdrant/bm42).
3. Batch processing: Configurable batch sizing with model reuse.
4. Failure tracking: Structured error reporting ensuring partial failures are not silently lost.
5. Telemetry & validation: Vector dimension checks and StageCounts telemetry integration.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import DimensionMismatchError, EmbedderError, ModelLoadError
from models import (
    Chunk,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingResult,
    FailedChunk,
    SparseEmbeddingData,
    StageCounts,
)

logger = logging.getLogger(__name__)


def _default_cache_dir() -> str:
    """Resolve the default absolute .fastembed_cache directory relative to repo root."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        return str(repo_root / ".fastembed_cache")
    except Exception:
        return "./.fastembed_cache"


class EmbeddingEngine:
    """Manages cached dense and sparse embedding model instances."""

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self.config = config if config is not None else EmbeddingConfig.from_env()
        self._dense_model: Any = None
        self._sparse_model: Any = None
        self._cache_dir = self.config.cache_dir or _default_cache_dir()

    def get_dense_model(self) -> Any:
        """Lazy-load and return the configured dense TextEmbedding model."""
        if self._dense_model is None:
            model_name = self.config.dense_model
            try:
                from fastembed import TextEmbedding

                logger.info("Loading dense embedding model: %s (cache: %s)", model_name, self._cache_dir)
                self._dense_model = TextEmbedding(
                    model_name=model_name,
                    cache_dir=self._cache_dir,
                )
            except Exception as exc:
                raise ModelLoadError(
                    source="embedder",
                    reason=f"Failed to load dense embedding model '{model_name}': {exc}",
                    model_name=model_name,
                ) from exc
        return self._dense_model

    def get_sparse_model(self) -> Any:
        """Lazy-load and return the configured sparse SparseTextEmbedding model."""
        if self._sparse_model is None:
            model_name = self.config.sparse_model
            if not model_name:
                return None
            try:
                from fastembed import SparseTextEmbedding

                logger.info("Loading sparse embedding model: %s (cache: %s)", model_name, self._cache_dir)
                self._sparse_model = SparseTextEmbedding(
                    model_name=model_name,
                    cache_dir=self._cache_dir,
                )
            except Exception as exc:
                raise ModelLoadError(
                    source="embedder",
                    reason=f"Failed to load sparse embedding model '{model_name}': {exc}",
                    model_name=model_name,
                ) from exc
        return self._sparse_model

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate dense embedding vectors for a batch of texts."""
        model = self.get_dense_model()
        try:
            embeddings = list(model.embed(texts))
            return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]
        except Exception as exc:
            raise EmbedderError(
                source="embedder",
                reason=f"Dense embedding generation failed: {exc}",
                model_name=self.config.dense_model,
            ) from exc

    def embed_sparse_batch(self, texts: list[str]) -> list[SparseEmbeddingData | None]:
        """Generate sparse embeddings for a batch of texts."""
        model = self.get_sparse_model()
        if model is None:
            return [None] * len(texts)
        try:
            raw_embeddings = list(model.embed(texts))
            sparse_list: list[SparseEmbeddingData | None] = []
            for item in raw_embeddings:
                if item is None:
                    sparse_list.append(None)
                elif hasattr(item, "indices") and hasattr(item, "values"):
                    indices = item.indices.tolist() if hasattr(item.indices, "tolist") else list(item.indices)
                    values = item.values.tolist() if hasattr(item.values, "tolist") else list(item.values)
                    sparse_list.append(SparseEmbeddingData(indices=indices, values=values))
                elif isinstance(item, dict):
                    sparse_list.append(
                        SparseEmbeddingData(
                            indices=list(item.get("indices", [])),
                            values=list(item.get("values", [])),
                        )
                    )
                else:
                    sparse_list.append(None)
            return sparse_list
        except Exception as exc:
            raise EmbedderError(
                source="embedder",
                reason=f"Sparse embedding generation failed: {exc}",
                model_name=self.config.sparse_model,
            ) from exc


def embed_chunks(
    chunks: list[Chunk] | Iterable[Chunk],
    config: EmbeddingConfig | None = None,
    *,
    engine: EmbeddingEngine | None = None,
    counts: StageCounts | None = None,
    strict: bool = False,
) -> EmbeddingResult:
    """Generate dense and sparse embeddings for a collection of Chunk objects.

    - Supports batch processing respecting config.batch_size.
    - Preserves deterministic Chunk ↔ Embedding relationships.
    - Validates dense embedding vector dimension against config.expected_dimension if specified.
    - On failure:
      - In strict=True mode: raises EmbedderError immediately.
      - In strict=False mode: captures failed chunks in result.failed_chunks with full context.
    - Updates StageCounts (embeddings_generated, chunks_failed) if provided.
    """
    chunk_list = list(chunks)
    counts = counts if counts is not None else StageCounts()
    config = config if config is not None else EmbeddingConfig.from_env()

    if not chunk_list:
        return EmbeddingResult(
            embedded_chunks=[],
            failed_chunks=[],
            total_chunks=0,
            successful_chunks=0,
            failed_count=0,
            batches_total=0,
            batches_successful=0,
            batches_failed=0,
        )

    # Validate chunk inputs
    for c in chunk_list:
        if not isinstance(c, Chunk):
            err = EmbedderError(
                source=getattr(c, "source_id", "unknown"),
                reason=f"Expected Chunk instance, got {type(c).__name__}",
                chunk_id=getattr(c, "chunk_id", None),
            )
            if strict:
                raise err
            else:
                counts.chunks_failed += 1
                return EmbeddingResult(
                    embedded_chunks=[],
                    failed_chunks=[FailedChunk(chunk=c, reason=str(err), error_type="TypeError")],  # type: ignore
                    total_chunks=len(chunk_list),
                    successful_chunks=0,
                    failed_count=len(chunk_list),
                    batches_total=1,
                    batches_successful=0,
                    batches_failed=1,
                )

    engine = engine if engine is not None else EmbeddingEngine(config)
    batch_size = max(1, config.batch_size)
    batches = [chunk_list[i : i + batch_size] for i in range(0, len(chunk_list), batch_size)]

    embedded_chunks: list[EmbeddedChunk] = []
    failed_chunks: list[FailedChunk] = []
    batches_successful = 0
    batches_failed = 0

    for batch in batches:
        batch_texts = [c.text for c in batch]
        try:
            dense_vectors = engine.embed_dense_batch(batch_texts)

            # Validate vector dimensions
            if config.expected_dimension is not None:
                for idx, dense_vec in enumerate(dense_vectors):
                    if len(dense_vec) != config.expected_dimension:
                        raise DimensionMismatchError(
                            source=batch[idx].source_id,
                            reason=(
                                f"Generated dense vector dimension ({len(dense_vec)}) "
                                f"does not match expected dimension ({config.expected_dimension})"
                            ),
                            chunk_id=batch[idx].chunk_id,
                            model_name=config.dense_model,
                        )

            sparse_vectors = engine.embed_sparse_batch(batch_texts)

            for chunk, dense_vec, sparse_vec in zip(batch, dense_vectors, sparse_vectors):
                embedded_chunks.append(
                    EmbeddedChunk(
                        chunk=chunk,
                        dense_embedding=dense_vec,
                        sparse_embedding=sparse_vec,
                    )
                )
            batches_successful += 1

        except Exception as exc:
            batches_failed += 1
            source_id = batch[0].source_id if batch else "unknown"
            chunk_id = batch[0].chunk_id if batch else None

            if isinstance(exc, EmbedderError):
                err = exc
            else:
                err = EmbedderError(
                    source=source_id,
                    reason=f"Batch embedding failed: {exc}",
                    chunk_id=chunk_id,
                    model_name=config.dense_model,
                )

            if strict:
                raise err from exc

            logger.error(str(getattr(err, "failure", err)))
            for chunk in batch:
                failed_chunks.append(
                    FailedChunk(
                        chunk=chunk,
                        reason=str(err),
                        error_type=type(exc).__name__,
                    )
                )

    counts.embeddings_generated += len(embedded_chunks)
    counts.chunks_failed += len(failed_chunks)

    logger.info(
        "Embedding completed: total=%d, successful=%d, failed=%d",
        len(chunk_list),
        len(embedded_chunks),
        len(failed_chunks),
    )

    return EmbeddingResult(
        embedded_chunks=embedded_chunks,
        failed_chunks=failed_chunks,
        total_chunks=len(chunk_list),
        successful_chunks=len(embedded_chunks),
        failed_count=len(failed_chunks),
        batches_total=len(batches),
        batches_successful=batches_successful,
        batches_failed=batches_failed,
    )


def embed_chunk(
    chunk: Chunk,
    config: EmbeddingConfig | None = None,
    *,
    engine: EmbeddingEngine | None = None,
) -> EmbeddedChunk:
    """Generate dense and sparse embeddings for a single Chunk object.

    Raises EmbedderError on failure.
    """
    result = embed_chunks([chunk], config=config, engine=engine, strict=True)
    if not result.embedded_chunks:
        raise EmbedderError(
            source=chunk.source_id,
            reason="Failed to generate embedding for chunk",
            chunk_id=chunk.chunk_id,
        )
    return result.embedded_chunks[0]
