"""Vector Store: EmbeddedChunk -> Qdrant hybrid vector collection.

Implements the Vector Store stage for the Aeronation-RAG data ingestion pipeline:
1. Collection management: Creates and validates Qdrant collections for hybrid retrieval (dense + sparse).
2. Non-destructive safety: Never automatically deletes, overwrites, or recreates existing collections.
3. Explicit client configuration: Requires valid Qdrant URL for production; prevents silent in-memory fallback.
4. Strict vector validation: Enforces dense vector dimension and sparse (indices, values) consistency.
5. Payload preservation: Preserves chunk text, source_id, chunk_id, page_num, and all metadata.
6. Batch upserting: Configurable batch insertion with deterministic UUID point IDs.
7. Failure tracking: Reports exact vectors inserted, failed, and batch statistics without silent drops.
"""

from __future__ import annotations

import logging
import re
import sys
import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import qdrant_client
import qdrant_client.models as qmodels

from errors import (
    CollectionNotFoundError,
    IncompatibleCollectionError,
    VectorDimensionError,
    VectorStoreError,
)
from models import (
    Chunk,
    EmbeddedChunk,
    FailedVector,
    StageCounts,
    VectorStoreConfig,
    VectorStoreResult,
)

logger = logging.getLogger(__name__)

# Sanitization regex: allows alphanumeric characters, underscores, and hyphens.
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def sanitize_collection_name(name: str) -> str:
    """Validate and sanitize a collection name.

    Only allows non-empty strings containing alphanumeric characters, underscores, and hyphens.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Collection name must be a non-empty string")
    trimmed = name.strip()
    if not _COLLECTION_NAME_RE.match(trimmed):
        raise ValueError(
            f"Invalid collection name '{trimmed}'. Only alphanumeric characters, underscores, and hyphens are permitted."
        )
    return trimmed


def _resolve_distance(distance_str: str) -> qmodels.Distance:
    """Resolve distance metric string to Qdrant Distance enum."""
    lookup = {
        "cosine": qmodels.Distance.COSINE,
        "dot": qmodels.Distance.DOT,
        "euclid": qmodels.Distance.EUCLID,
        "manhattan": qmodels.Distance.MANHATTAN,
    }
    key = (distance_str or "cosine").strip().lower()
    if key not in lookup:
        raise ValueError(f"Unsupported distance metric '{distance_str}'. Supported: {list(lookup.keys())}")
    return lookup[key]


def derive_point_id(chunk_id: str) -> str:
    """Derive a deterministic, valid UUID string for a Qdrant point from chunk_id."""
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(f"Invalid chunk_id for point ID derivation: {chunk_id!r}")
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"urn:chunk:{chunk_id.strip()}"))


def build_point_payload(chunk: Chunk) -> dict[str, Any]:
    """Build the Qdrant point payload dictionary preserving all chunk information and metadata."""
    payload: dict[str, Any] = {
        "text": chunk.text,
        "source_id": chunk.source_id,
        "chunk_id": chunk.chunk_id,
        "page_num": chunk.page_num,
    }
    if chunk.metadata:
        payload["metadata"] = dict(chunk.metadata)
        for k, v in chunk.metadata.items():
            if k not in payload:
                payload[k] = v
    return payload


def get_qdrant_client(
    config: VectorStoreConfig | None = None,
    client: Any | None = None,
    *,
    allow_in_memory: bool = False,
) -> qdrant_client.QdrantClient:
    """Resolve or instantiate a QdrantClient instance.

    Requires a valid Qdrant URL in production. Missing configuration raises a VectorStoreError
    unless allow_in_memory=True is explicitly specified (e.g. in test suites).
    """
    if client is not None:
        return client

    cfg = config if config is not None else VectorStoreConfig.from_env()
    if cfg.url:
        return qdrant_client.QdrantClient(url=cfg.url, api_key=cfg.api_key)

    if allow_in_memory:
        logger.warning("Instantiating test-only in-memory QdrantClient (location=':memory:').")
        return qdrant_client.QdrantClient(location=":memory:")

    col_name = cfg.collection_name if cfg and cfg.collection_name else "unknown"
    raise VectorStoreError(
        collection=col_name,
        reason=(
            "Qdrant URL is not configured (missing QDRANT_URL environment variable or config). "
            "Ingestion cannot proceed without a valid Qdrant server connection."
        ),
    )


def validate_collection(client: Any, config: VectorStoreConfig) -> bool:
    """Validate that an existing Qdrant collection is compatible with config.

    Verifies existence, dense vector dimension, distance metric, and sparse vector config.
    Raises IncompatibleCollectionError or CollectionNotFoundError on failure.
    """
    col_name = sanitize_collection_name(config.collection_name)
    try:
        exists = client.collection_exists(col_name)
    except Exception as exc:
        raise VectorStoreError(collection=col_name, reason=f"Failed checking collection existence: {exc}") from exc

    if not exists:
        raise CollectionNotFoundError(
            collection=col_name,
            reason=f"Collection '{col_name}' does not exist in Qdrant",
        )

    try:
        col_info = client.get_collection(col_name)
    except Exception as exc:
        raise VectorStoreError(collection=col_name, reason=f"Failed getting collection details: {exc}") from exc

    params = getattr(col_info.config, "params", None) if hasattr(col_info, "config") else None
    if params is None:
        return True

    # 1. Validate dense vectors configuration
    vectors = getattr(params, "vectors", None)
    dense_param = None

    if isinstance(vectors, dict):
        if config.dense_vector_name in vectors:
            dense_param = vectors[config.dense_vector_name]
        else:
            raise IncompatibleCollectionError(
                collection=col_name,
                reason=f"Dense vector '{config.dense_vector_name}' not found in collection vectors: {list(vectors.keys())}",
            )
    elif isinstance(vectors, qmodels.VectorParams):
        dense_param = vectors

    if dense_param is not None:
        actual_dim = getattr(dense_param, "size", None)
        if actual_dim is not None and actual_dim != config.dimension:
            raise IncompatibleCollectionError(
                collection=col_name,
                reason=f"Dense vector dimension mismatch: collection has {actual_dim}, but config requires {config.dimension}",
            )

        actual_dist = getattr(dense_param, "distance", None)
        expected_dist = _resolve_distance(config.distance)
        if actual_dist is not None:
            actual_dist_val = actual_dist.value if hasattr(actual_dist, "value") else str(actual_dist)
            expected_dist_val = expected_dist.value if hasattr(expected_dist, "value") else str(expected_dist)
            if actual_dist_val.lower() != expected_dist_val.lower():
                raise IncompatibleCollectionError(
                    collection=col_name,
                    reason=f"Distance metric mismatch: collection has '{actual_dist_val}', but config requires '{expected_dist_val}'",
                )

    # 2. Validate sparse vectors configuration if hybrid is enabled
    if config.enable_hybrid:
        sparse_vectors = getattr(params, "sparse_vectors", None)
        if not sparse_vectors:
            raise IncompatibleCollectionError(
                collection=col_name,
                reason=f"Collection '{col_name}' lacks sparse vectors configuration required for hybrid retrieval",
            )
        if isinstance(sparse_vectors, dict) and config.sparse_vector_name not in sparse_vectors:
            raise IncompatibleCollectionError(
                collection=col_name,
                reason=f"Sparse vector '{config.sparse_vector_name}' not found in sparse vectors: {list(sparse_vectors.keys())}",
            )

    return True


def create_collection(
    client: Any,
    config: VectorStoreConfig,
) -> bool:
    """Create a new Qdrant collection or verify compatibility of an existing one.

    - If the collection exists: validates that dimensions, distance metric, and hybrid sparse configuration are compatible. Incompatibilities raise IncompatibleCollectionError.
    - If the collection does not exist: creates a new collection using client.create_collection.
    - Never automatically deletes, overwrites, or recreates an existing collection.
    """
    col_name = sanitize_collection_name(config.collection_name)
    expected_dist = _resolve_distance(config.distance)

    try:
        exists = client.collection_exists(col_name)
    except Exception:
        exists = False

    if exists:
        logger.info("Collection '%s' already exists; verifying compatibility...", col_name)
        validate_collection(client, config)
        return True

    logger.info(
        "Creating Qdrant collection '%s' (dim=%d, distance=%s, hybrid=%s)...",
        col_name,
        config.dimension,
        config.distance,
        config.enable_hybrid,
    )

    dense_params = qmodels.VectorParams(
        size=config.dimension,
        distance=expected_dist,
    )

    if config.enable_hybrid:
        vectors_config = {config.dense_vector_name: dense_params}
        sparse_vectors_config = {config.sparse_vector_name: qmodels.SparseVectorParams()}
    else:
        vectors_config = {config.dense_vector_name: dense_params}
        sparse_vectors_config = None

    try:
        client.create_collection(
            collection_name=col_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
    except Exception as exc:
        raise VectorStoreError(
            collection=col_name,
            reason=f"Failed to create Qdrant collection: {exc}",
        ) from exc

    return True


def upsert_embeddings(
    client: Any,
    embedded_chunks: list[EmbeddedChunk] | Iterable[EmbeddedChunk],
    config: VectorStoreConfig | None = None,
    *,
    counts: StageCounts | None = None,
    strict: bool = False,
) -> VectorStoreResult:
    """Batch upsert embedded chunks into the configured Qdrant collection.

    - Validates dense embedding existence and dimension for every chunk.
    - Validates sparse vector format (length match, numeric values) if hybrid search is enabled.
    - Preserves deterministic UUID point IDs and full chunk payload.
    - Updates StageCounts (vectors_inserted, vectors_failed) if provided.
    - On failure, captures failed points in result.failed_vectors (or raises VectorStoreError in strict mode).
    """
    items = list(embedded_chunks)
    counts = counts if counts is not None else StageCounts()
    config = config if config is not None else VectorStoreConfig.from_env()
    col_name = sanitize_collection_name(config.collection_name)

    if not items:
        return VectorStoreResult(
            collection_name=col_name,
            vectors_expected=0,
            vectors_inserted=0,
            vectors_failed=0,
            batches_total=0,
            batches_successful=0,
            batches_failed=0,
        )

    batch_size = max(1, config.batch_size)
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    inserted_count = 0
    failed_vectors: list[FailedVector] = []
    batches_successful = 0
    batches_failed = 0

    for batch in batches:
        points_batch: list[qmodels.PointStruct] = []
        batch_valid = True

        for item in batch:
            if not isinstance(item, EmbeddedChunk) or not isinstance(item.chunk, Chunk):
                err_msg = f"Expected EmbeddedChunk instance, got {type(item).__name__}"
                if strict:
                    raise VectorStoreError(collection=col_name, reason=err_msg)
                failed_vectors.append(
                    FailedVector(
                        chunk_id=getattr(getattr(item, "chunk", None), "chunk_id", "unknown"),
                        source_id=getattr(getattr(item, "chunk", None), "source_id", "unknown"),
                        reason=err_msg,
                        error_type="TypeError",
                    )
                )
                batch_valid = False
                continue

            chunk = item.chunk

            # 1. Validate dense vector
            if not item.dense_embedding or not isinstance(item.dense_embedding, (list, tuple)):
                err_msg = f"Missing or invalid dense embedding on chunk '{chunk.chunk_id}'"
                if strict:
                    raise VectorDimensionError(collection=col_name, reason=err_msg, chunk_id=chunk.chunk_id, source_id=chunk.source_id)
                failed_vectors.append(FailedVector(chunk_id=chunk.chunk_id, source_id=chunk.source_id, reason=err_msg, error_type="DenseEmbeddingError"))
                batch_valid = False
                continue

            if len(item.dense_embedding) != config.dimension:
                err_msg = f"Dense vector dimension mismatch on chunk '{chunk.chunk_id}': expected {config.dimension}, got {len(item.dense_embedding)}"
                if strict:
                    raise VectorDimensionError(collection=col_name, reason=err_msg, chunk_id=chunk.chunk_id, source_id=chunk.source_id)
                failed_vectors.append(FailedVector(chunk_id=chunk.chunk_id, source_id=chunk.source_id, reason=err_msg, error_type="DimensionMismatch"))
                batch_valid = False
                continue

            # 2. Validate sparse vector if hybrid search is enabled
            if config.enable_hybrid:
                sparse_emb = item.sparse_embedding
                if (
                    sparse_emb is None
                    or not hasattr(sparse_emb, "indices")
                    or not hasattr(sparse_emb, "values")
                ):
                    err_msg = f"Missing or invalid sparse embedding on chunk '{chunk.chunk_id}' for hybrid collection"
                    if strict:
                        raise VectorStoreError(collection=col_name, reason=err_msg, chunk_id=chunk.chunk_id, source_id=chunk.source_id)
                    failed_vectors.append(FailedVector(chunk_id=chunk.chunk_id, source_id=chunk.source_id, reason=err_msg, error_type="SparseEmbeddingError"))
                    batch_valid = False
                    continue

                indices = sparse_emb.indices
                values = sparse_emb.values

                if not isinstance(indices, (list, tuple)) or not isinstance(values, (list, tuple)):
                    err_msg = f"Sparse embedding indices and values must be lists/sequences on chunk '{chunk.chunk_id}'"
                    if strict:
                        raise VectorStoreError(collection=col_name, reason=err_msg, chunk_id=chunk.chunk_id, source_id=chunk.source_id)
                    failed_vectors.append(FailedVector(chunk_id=chunk.chunk_id, source_id=chunk.source_id, reason=err_msg, error_type="SparseEmbeddingError"))
                    batch_valid = False
                    continue

                if len(indices) != len(values):
                    err_msg = (
                        f"Sparse embedding indices and values length mismatch on chunk '{chunk.chunk_id}': "
                        f"len(indices)={len(indices)} vs len(values)={len(values)}"
                    )
                    if strict:
                        raise VectorStoreError(collection=col_name, reason=err_msg, chunk_id=chunk.chunk_id, source_id=chunk.source_id)
                    failed_vectors.append(FailedVector(chunk_id=chunk.chunk_id, source_id=chunk.source_id, reason=err_msg, error_type="SparseEmbeddingError"))
                    batch_valid = False
                    continue

                if not all(isinstance(idx, int) for idx in indices) or not all(isinstance(v, (int, float)) for v in values):
                    err_msg = f"Sparse embedding contains non-numeric indices or values on chunk '{chunk.chunk_id}'"
                    if strict:
                        raise VectorStoreError(collection=col_name, reason=err_msg, chunk_id=chunk.chunk_id, source_id=chunk.source_id)
                    failed_vectors.append(FailedVector(chunk_id=chunk.chunk_id, source_id=chunk.source_id, reason=err_msg, error_type="SparseEmbeddingError"))
                    batch_valid = False
                    continue

            # 3. Construct point
            point_id = derive_point_id(chunk.chunk_id)
            payload = build_point_payload(chunk)

            if config.enable_hybrid and item.sparse_embedding is not None:
                vector_payload = {
                    config.dense_vector_name: item.dense_embedding,
                    config.sparse_vector_name: qmodels.SparseVector(
                        indices=list(item.sparse_embedding.indices),
                        values=list(item.sparse_embedding.values),
                    ),
                }
            else:
                vector_payload = {config.dense_vector_name: item.dense_embedding}

            points_batch.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector_payload,
                    payload=payload,
                )
            )

        if not batch_valid and strict:
            break

        if points_batch:
            try:
                client.upsert(
                    collection_name=col_name,
                    points=points_batch,
                )
                inserted_count += len(points_batch)
                batches_successful += 1
            except Exception as exc:
                batches_failed += 1
                err = VectorStoreError(
                    collection=col_name,
                    reason=f"Qdrant upsert batch failed: {exc}",
                )
                if strict:
                    raise err from exc
                logger.error(str(err.failure))
                for pt in points_batch:
                    cid = pt.payload.get("chunk_id", "unknown") if pt.payload else "unknown"
                    sid = pt.payload.get("source_id", "unknown") if pt.payload else "unknown"
                    failed_vectors.append(
                        FailedVector(
                            chunk_id=cid,
                            source_id=sid,
                            reason=str(err),
                            error_type=type(exc).__name__,
                        )
                    )

    counts.vectors_inserted += inserted_count
    counts.vectors_failed += len(failed_vectors)

    logger.info(
        "VectorStore upsert completed for '%s': expected=%d, inserted=%d, failed=%d",
        col_name,
        len(items),
        inserted_count,
        len(failed_vectors),
    )

    return VectorStoreResult(
        collection_name=col_name,
        vectors_expected=len(items),
        vectors_inserted=inserted_count,
        vectors_failed=len(failed_vectors),
        failed_vectors=failed_vectors,
        batches_total=len(batches),
        batches_successful=batches_successful,
        batches_failed=batches_failed,
    )


def get_collection_info(client: Any, collection_name: str) -> dict[str, Any]:
    """Retrieve collection status, point counts, and configuration summary."""
    col_name = sanitize_collection_name(collection_name)
    try:
        info = client.get_collection(col_name)
        return {
            "collection_name": col_name,
            "status": getattr(info, "status", "ok"),
            "vectors_count": getattr(info, "vectors_count", getattr(info, "points_count", 0)),
            "points_count": getattr(info, "points_count", 0),
        }
    except Exception as exc:
        raise VectorStoreError(collection=col_name, reason=f"Failed retrieving collection info: {exc}") from exc
