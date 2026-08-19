"""Shared data models for the ingestion pipeline (loader, cleaner, chunker, embedder, vector_store, persistence, etc).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RawDoc:
    """A raw or cleaned document/page representation in the ingestion pipeline."""

    text: str
    source_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StageCounts:
    """Telemetry counts across ingestion pipeline stages."""

    files_seen: int = 0
    docs_loaded: int = 0
    docs_cleaned: int = 0
    docs_discarded: int = 0
    chunks_created: int = 0
    embeddings_generated: int = 0
    chunks_failed: int = 0
    vectors_inserted: int = 0
    vectors_failed: int = 0


@dataclass
class Chunk:
    """Chunk model produced by Chunker stage."""

    chunk_id: str
    text: str
    source_id: str
    page_num: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingConfig:
    """Configuration for chunking stage, configurable via environment variables or config dict."""

    chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("RAG_CHUNK_SIZE")
            or os.getenv("RAG_CITATION_CHUNK_SIZE")
            or os.getenv("CHUNK_SIZE")
            or 512
        )
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(
            os.getenv("RAG_CHUNK_OVERLAP")
            or os.getenv("RAG_CITATION_CHUNK_OVERLAP")
            or os.getenv("CHUNK_OVERLAP")
            or 64
        )
    )
    min_chunk_size: int = field(
        default_factory=lambda: int(
            os.getenv("RAG_MIN_CHUNK_SIZE")
            or os.getenv("MIN_CHUNK_SIZE")
            or 0
        )
    )

    @classmethod
    def from_env(cls, config_dict: dict | None = None) -> ChunkingConfig:
        """Construct ChunkingConfig from environment variables with fallback to config dict."""
        cfg = config_dict or {}
        size = int(
            os.getenv("RAG_CHUNK_SIZE")
            or os.getenv("CHUNK_SIZE")
            or cfg.get("RAG_CHUNK_SIZE")
            or cfg.get("RAG_CITATION_CHUNK_SIZE")
            or 512
        )
        overlap = int(
            os.getenv("RAG_CHUNK_OVERLAP")
            or os.getenv("CHUNK_OVERLAP")
            or cfg.get("RAG_CHUNK_OVERLAP")
            or cfg.get("RAG_CITATION_CHUNK_OVERLAP")
            or 64
        )
        min_size = int(
            os.getenv("RAG_MIN_CHUNK_SIZE")
            or os.getenv("MIN_CHUNK_SIZE")
            or cfg.get("RAG_MIN_CHUNK_SIZE")
            or 0
        )
        return cls(chunk_size=size, chunk_overlap=overlap, min_chunk_size=min_size)


@dataclass
class SparseEmbeddingData:
    """Sparse vector representation containing non-zero token indices and weights."""

    indices: list[int]
    values: list[float]

    def as_dict(self) -> dict[str, list[Any]]:
        return {"indices": self.indices, "values": self.values}


@dataclass
class EmbeddedChunk:
    """A chunk enriched with dense and sparse vector embeddings."""

    chunk: Chunk
    dense_embedding: list[float]
    sparse_embedding: SparseEmbeddingData | None = None


@dataclass
class FailedChunk:
    """Record of a chunk that failed during the embedding stage."""

    chunk: Chunk
    reason: str
    error_type: str = "EmbeddingError"


@dataclass
class EmbeddingResult:
    """Summary and artifacts of an embedding operation."""

    embedded_chunks: list[EmbeddedChunk]
    failed_chunks: list[FailedChunk] = field(default_factory=list)
    total_chunks: int = 0
    successful_chunks: int = 0
    failed_count: int = 0
    batches_total: int = 0
    batches_successful: int = 0
    batches_failed: int = 0


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models and batch processing."""

    dense_model: str = field(
        default_factory=lambda: os.getenv("HF_EMBED")
        or os.getenv("EMBEDDING_MODEL")
        or os.getenv("DENSE_EMBEDDING_MODEL")
        or "sentence-transformers/all-MiniLM-L6-v2"
    )
    sparse_model: str | None = field(
        default_factory=lambda: os.getenv("FASTEMBED_SPARSE_MODEL")
        or os.getenv("SPARSE_EMBEDDING_MODEL")
        or "Qdrant/bm42-all-minilm-l6-v2-attentions"
    )
    batch_size: int = field(
        default_factory=lambda: int(
            os.getenv("EMBEDDING_BATCH_SIZE")
            or os.getenv("RAG_EMBEDDING_BATCH_SIZE")
            or os.getenv("BATCH_SIZE")
            or 32
        )
    )
    expected_dimension: int | None = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION"))
        if os.getenv("EMBEDDING_DIMENSION")
        else None
    )
    cache_dir: str | None = field(
        default_factory=lambda: os.getenv("FASTEMBED_CACHE_DIR")
    )

    @classmethod
    def from_env(cls, config_dict: dict | None = None) -> EmbeddingConfig:
        """Construct EmbeddingConfig from environment variables with fallback to config dict."""
        cfg = config_dict or {}
        dense = (
            os.getenv("HF_EMBED")
            or os.getenv("EMBEDDING_MODEL")
            or os.getenv("DENSE_EMBEDDING_MODEL")
            or cfg.get("HF_EMBED")
            or cfg.get("EMBEDDING_MODEL")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        sparse = (
            os.getenv("FASTEMBED_SPARSE_MODEL")
            or os.getenv("SPARSE_EMBEDDING_MODEL")
            or cfg.get("FASTEMBED_SPARSE_MODEL")
            or cfg.get("SPARSE_EMBEDDING_MODEL")
            or "Qdrant/bm42-all-minilm-l6-v2-attentions"
        )
        b_size = int(
            os.getenv("EMBEDDING_BATCH_SIZE")
            or os.getenv("RAG_EMBEDDING_BATCH_SIZE")
            or os.getenv("BATCH_SIZE")
            or cfg.get("EMBEDDING_BATCH_SIZE")
            or 32
        )
        exp_dim_val = (
            os.getenv("EMBEDDING_DIMENSION")
            or cfg.get("EMBEDDING_DIMENSION")
            or cfg.get("DENSE_DIMENSION")
        )
        exp_dim = int(exp_dim_val) if exp_dim_val is not None else None
        c_dir = (
            os.getenv("FASTEMBED_CACHE_DIR")
            or cfg.get("FASTEMBED_CACHE_DIR")
        )
        return cls(
            dense_model=dense,
            sparse_model=sparse,
            batch_size=b_size,
            expected_dimension=exp_dim,
            cache_dir=c_dir,
        )


@dataclass
class FailedVector:
    """Record of a point/vector that failed during vector store operations."""

    chunk_id: str
    source_id: str
    reason: str
    error_type: str = "VectorStoreError"


@dataclass
class VectorStoreResult:
    """Result summary of a vector store upsert operation."""

    collection_name: str
    vectors_expected: int = 0
    vectors_inserted: int = 0
    vectors_failed: int = 0
    failed_vectors: list[FailedVector] = field(default_factory=list)
    batches_total: int = 0
    batches_successful: int = 0
    batches_failed: int = 0


@dataclass
class VectorStoreConfig:
    """Configuration for Qdrant collection creation, validation, and vector upserting."""

    collection_name: str = field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION") or "rag_llm"
    )
    dimension: int = field(
        default_factory=lambda: int(
            os.getenv("VECTOR_DIMENSION")
            or os.getenv("EMBEDDING_DIMENSION")
            or 384
        )
    )
    distance: str = field(
        default_factory=lambda: os.getenv("VECTOR_DISTANCE")
        or os.getenv("QDRANT_DISTANCE")
        or "Cosine"
    )
    enable_hybrid: bool = field(
        default_factory=lambda: os.getenv("QDRANT_ENABLE_HYBRID", "True").lower() in ("true", "1", "yes")
    )
    dense_vector_name: str = field(
        default_factory=lambda: os.getenv("DENSE_VECTOR_NAME") or "text-dense"
    )
    sparse_vector_name: str = field(
        default_factory=lambda: os.getenv("SPARSE_VECTOR_NAME") or "text-sparse-new"
    )
    batch_size: int = field(
        default_factory=lambda: int(
            os.getenv("VECTOR_BATCH_SIZE")
            or os.getenv("QDRANT_BATCH_SIZE")
            or 64
        )
    )
    url: str | None = field(
        default_factory=lambda: os.getenv("QDRANT_URL")
    )
    api_key: str | None = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY")
    )

    @classmethod
    def from_env(cls, config_dict: dict | None = None, collection_name: str | None = None) -> VectorStoreConfig:
        """Construct VectorStoreConfig from environment variables with fallback to config dict."""
        cfg = config_dict or {}
        col = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or cfg.get("QDRANT_COLLECTION")
            or "rag_llm"
        )
        dim = int(
            os.getenv("VECTOR_DIMENSION")
            or os.getenv("EMBEDDING_DIMENSION")
            or cfg.get("VECTOR_DIMENSION")
            or cfg.get("EMBEDDING_DIMENSION")
            or 384
        )
        dist = (
            os.getenv("VECTOR_DISTANCE")
            or os.getenv("QDRANT_DISTANCE")
            or cfg.get("VECTOR_DISTANCE")
            or cfg.get("QDRANT_DISTANCE")
            or "Cosine"
        )
        hybrid = (
            os.getenv("QDRANT_ENABLE_HYBRID", "").lower() in ("true", "1", "yes")
            if os.getenv("QDRANT_ENABLE_HYBRID") is not None
            else bool(cfg.get("QDRANT_ENABLE_HYBRID", True))
        )
        dense_name = (
            os.getenv("DENSE_VECTOR_NAME")
            or cfg.get("DENSE_VECTOR_NAME")
            or "text-dense"
        )
        sparse_name = (
            os.getenv("SPARSE_VECTOR_NAME")
            or cfg.get("SPARSE_VECTOR_NAME")
            or "text-sparse-new"
        )
        b_size = int(
            os.getenv("VECTOR_BATCH_SIZE")
            or os.getenv("QDRANT_BATCH_SIZE")
            or cfg.get("VECTOR_BATCH_SIZE")
            or cfg.get("QDRANT_BATCH_SIZE")
            or 64
        )
        q_url = (
            os.getenv("QDRANT_URL")
            or cfg.get("QDRANT_URL")
        )
        q_key = (
            os.getenv("QDRANT_API_KEY")
            or cfg.get("QDRANT_API_KEY")
        )
        return cls(
            collection_name=col,
            dimension=dim,
            distance=dist,
            enable_hybrid=hybrid,
            dense_vector_name=dense_name,
            sparse_vector_name=sparse_name,
            batch_size=b_size,
            url=q_url,
            api_key=q_key,
        )


@dataclass
class PersistenceConfig:
    """Configuration for local persistence directory, versioning, and manifest generation."""

    collection_type: str = field(
        default_factory=lambda: os.getenv("PERSIST_COLLECTION_TYPE") or "rag_llm"
    )
    ingestion_id: str | None = field(
        default_factory=lambda: os.getenv("INGESTION_ID")
    )
    base_persist_dir: str = field(
        default_factory=lambda: os.getenv("PERSIST_DIR") or "persist"
    )

    @classmethod
    def from_env(cls, config_dict: dict | None = None, collection_type: str | None = None) -> PersistenceConfig:
        """Construct PersistenceConfig from environment variables with fallback to config dict."""
        cfg = config_dict or {}
        col = (
            collection_type
            or os.getenv("PERSIST_COLLECTION_TYPE")
            or os.getenv("QDRANT_COLLECTION")
            or cfg.get("PERSIST_COLLECTION_TYPE")
            or cfg.get("QDRANT_COLLECTION")
            or "rag_llm"
        )
        ing_id = (
            os.getenv("INGESTION_ID")
            or cfg.get("INGESTION_ID")
        )
        base_dir = (
            os.getenv("PERSIST_DIR")
            or cfg.get("PERSIST_DIR")
            or "persist"
        )
        return cls(collection_type=col, ingestion_id=ing_id, base_persist_dir=base_dir)


@dataclass
class PersistenceResult:
    """Result summary of a persistence operation."""

    collection_type: str
    ingestion_id: str
    persist_dir: str
    manifest_path: str
    files: list[str]
    checksums: dict[str, str]
    success: bool = True


class IngestionStatus:
    """Standard lifecycle statuses for an ingestion execution attempt."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    FAILED = "FAILED"
    VALIDATING = "VALIDATING"
    UPLOADING = "UPLOADING"
    PROMOTING = "PROMOTING"
    COMPLETED = "COMPLETED"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"


@dataclass
class IngestionState:
    """Durable state tracker for an ingestion execution and its retry lifecycle."""

    ingestion_id: str
    version: str
    collection_type: str = "rag_llm"
    status: str = IngestionStatus.PENDING
    current_stage: str = "init"
    failed_stage: str | None = None
    attempt: int = 1
    max_attempts: int = 3
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def can_retry(self) -> bool:
        """Check whether another retry attempt is permitted."""
        return self.attempt < self.max_attempts and self.status not in (
            IngestionStatus.COMPLETED,
            IngestionStatus.RETRY_EXHAUSTED,
        )

    def mark_stage(self, stage: str, status: str = IngestionStatus.PROCESSING) -> None:
        """Update current stage and status."""
        self.current_stage = stage
        self.status = status
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_failed(self, stage: str, error: str) -> None:
        """Mark attempt as failed at a specific stage."""
        self.failed_stage = stage
        self.current_stage = stage
        self.error = error
        self.status = IngestionStatus.FAILED
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_retry_exhausted(self, stage: str, error: str) -> None:
        """Mark pipeline as having exhausted all configured retry attempts."""
        self.failed_stage = stage
        self.current_stage = stage
        self.error = error
        self.status = IngestionStatus.RETRY_EXHAUSTED
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def mark_completed(self) -> None:
        """Mark ingestion as fully completed and validated."""
        self.current_stage = "completed"
        self.status = IngestionStatus.COMPLETED
        self.error = None
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "ingestion_id": self.ingestion_id,
            "version": self.version,
            "collection_type": self.collection_type,
            "status": self.status,
            "current_stage": self.current_stage,
            "failed_stage": self.failed_stage,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IngestionState:
        """Construct from dictionary."""
        return cls(
            ingestion_id=data.get("ingestion_id", "unknown"),
            version=data.get("version", data.get("ingestion_id", "unknown")),
            collection_type=data.get("collection_type", "rag_llm"),
            status=data.get("status", IngestionStatus.PENDING),
            current_stage=data.get("current_stage", "init"),
            failed_stage=data.get("failed_stage"),
            attempt=int(data.get("attempt", 1)),
            max_attempts=int(data.get("max_attempts", 3)),
            error=data.get("error"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PipelineConfig:
    """Top-level configuration for pipeline execution and retry orchestration."""

    collection_type: str = field(
        default_factory=lambda: os.getenv("PERSIST_COLLECTION_TYPE") or os.getenv("QDRANT_COLLECTION") or "rag_llm"
    )
    ingestion_id: str | None = field(
        default_factory=lambda: os.getenv("INGESTION_ID")
    )
    max_attempts: int = field(
        default_factory=lambda: int(
            os.getenv("MAX_REINGESTION_ATTEMPTS")
            or os.getenv("RAG_MAX_REINGESTION_ATTEMPTS")
            or os.getenv("MAX_RETRIES")
            or 3
        )
    )
    data_dir: str | None = field(default_factory=lambda: os.getenv("DATA_DIR") or "data")
    urls: list[str] | None = None
    persist_dir: str = field(default_factory=lambda: os.getenv("PERSIST_DIR") or "persist")
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig.from_env)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig.from_env)
    vector_config: VectorStoreConfig = field(default_factory=VectorStoreConfig.from_env)
    persistence_config: PersistenceConfig = field(default_factory=PersistenceConfig.from_env)

    @classmethod
    def from_env(cls, config_dict: dict | None = None, collection_type: str | None = None) -> PipelineConfig:
        cfg = config_dict or {}
        col = (
            collection_type
            or os.getenv("PERSIST_COLLECTION_TYPE")
            or os.getenv("QDRANT_COLLECTION")
            or cfg.get("PERSIST_COLLECTION_TYPE")
            or cfg.get("QDRANT_COLLECTION")
            or "rag_llm"
        )
        ing_id = os.getenv("INGESTION_ID") or cfg.get("INGESTION_ID")
        max_att = int(
            os.getenv("MAX_REINGESTION_ATTEMPTS")
            or os.getenv("RAG_MAX_REINGESTION_ATTEMPTS")
            or os.getenv("MAX_RETRIES")
            or cfg.get("MAX_REINGESTION_ATTEMPTS")
            or 3
        )
        d_dir = os.getenv("DATA_DIR") or cfg.get("DATA_DIR") or "data"
        p_dir = os.getenv("PERSIST_DIR") or cfg.get("PERSIST_DIR") or "persist"

        return cls(
            collection_type=col,
            ingestion_id=ing_id,
            max_attempts=max_att,
            data_dir=d_dir,
            persist_dir=p_dir,
            chunking_config=ChunkingConfig.from_env(cfg),
            embedding_config=EmbeddingConfig.from_env(cfg),
            vector_config=VectorStoreConfig.from_env(cfg, collection_name=col),
            persistence_config=PersistenceConfig.from_env(cfg, collection_type=col),
        )


@dataclass
class PipelineResult:
    """Overall summary and telemetry of an ingestion pipeline run."""

    ingestion_id: str
    collection_type: str
    status: str
    attempt: int
    max_attempts: int
    counts: StageCounts = field(default_factory=StageCounts)
    persistence_result: PersistenceResult | None = None
    vector_result: VectorStoreResult | None = None
    error: str | None = None
    success: bool = False


@dataclass
class S3UploadConfig:
    """Configuration for S3 persistence artifact upload and verification."""

    bucket_name: str = field(
        default_factory=lambda: os.getenv("S3_PERSIST_BUCKET")
        or os.getenv("S3_BUCKET")
        or ""
    )
    base_prefix: str = field(
        default_factory=lambda: os.getenv("S3_PERSIST_PREFIX")
        or os.getenv("PERSIST_PREFIX")
        or "persist"
    )
    region_name: str = field(
        default_factory=lambda: os.getenv("AWS_REGION")
        or "ap-south-1"
    )
    staging_prefix: str = ".tmp"
    enable_staging: bool = True
    overwrite: bool = False
    collection_type: str | None = None

    @classmethod
    def from_env(
        cls,
        config_dict: dict | None = None,
        collection_type: str | None = None,
    ) -> S3UploadConfig:
        """Construct S3UploadConfig from environment variables with fallback to config dict."""
        cfg = config_dict or {}
        bucket = (
            os.getenv("S3_PERSIST_BUCKET")
            or os.getenv("S3_BUCKET")
            or cfg.get("S3_PERSIST_BUCKET")
            or cfg.get("S3_LOGS_BUCKET")
            or cfg.get("S3_BUCKET")
            or ""
        )
        base = (
            os.getenv("S3_PERSIST_PREFIX")
            or os.getenv("PERSIST_PREFIX")
            or cfg.get("S3_PERSIST_PREFIX")
            or cfg.get("PERSIST_PREFIX")
            or "persist"
        )
        region = (
            os.getenv("AWS_REGION")
            or cfg.get("AWS_REGION")
            or "ap-south-1"
        )
        col = (
            collection_type
            or os.getenv("PERSIST_COLLECTION_TYPE")
            or os.getenv("COLLECTION_TYPE")
            or cfg.get("PERSIST_COLLECTION_TYPE")
            or cfg.get("COLLECTION_TYPE")
        )
        staging = cfg.get("S3_STAGING_PREFIX", ".tmp")
        enable_stg = (
            os.getenv("S3_ENABLE_STAGING", "True").lower() in ("true", "1", "yes")
            if os.getenv("S3_ENABLE_STAGING") is not None
            else bool(cfg.get("S3_ENABLE_STAGING", True))
        )
        ow = (
            os.getenv("S3_OVERWRITE", "False").lower() in ("true", "1", "yes")
            if os.getenv("S3_OVERWRITE") is not None
            else bool(cfg.get("S3_OVERWRITE", False))
        )

        return cls(
            bucket_name=bucket,
            base_prefix=base,
            region_name=region,
            staging_prefix=staging,
            enable_staging=enable_stg,
            overwrite=ow,
            collection_type=col,
        )


@dataclass
class S3ObjectInfo:
    """Metadata for an object verified in S3."""

    key: str
    size: int = 0
    etag: str | None = None
    last_modified: str | None = None


@dataclass
class S3UploadResult:
    """Telemetry and outcome of an S3 upload operation."""

    bucket: str
    prefix: str
    collection_type: str
    ingestion_id: str
    files_expected: int = 0
    files_uploaded: int = 0
    files_failed: int = 0
    bytes_uploaded: int = 0
    uploaded_keys: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    manifest_key: str = ""
    success: bool = False
    error: str | None = None


@dataclass
class UserPipelineConfig:
    """Configuration for single-request user ingestion pipeline execution."""

    collection_type: str = field(
        default_factory=lambda: os.getenv("PERSIST_COLLECTION_TYPE")
        or os.getenv("QDRANT_COLLECTION")
        or "rag_llm"
    )
    ingestion_id: str | None = field(
        default_factory=lambda: os.getenv("INGESTION_ID")
    )
    persist_dir: str = field(
        default_factory=lambda: os.getenv("PERSIST_DIR") or "persist"
    )
    persist_artifact: bool = True
    upload_to_s3: bool = False
    chunking_config: ChunkingConfig = field(default_factory=ChunkingConfig.from_env)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig.from_env)
    vector_config: VectorStoreConfig = field(default_factory=VectorStoreConfig.from_env)
    persistence_config: PersistenceConfig = field(default_factory=PersistenceConfig.from_env)
    s3_config: S3UploadConfig | None = None

    @classmethod
    def from_env(
        cls,
        config_dict: dict | None = None,
        collection_type: str | None = None,
    ) -> UserPipelineConfig:
        """Construct UserPipelineConfig from environment variables with fallback to config dict."""
        cfg = config_dict or {}
        col = (
            collection_type
            or os.getenv("PERSIST_COLLECTION_TYPE")
            or os.getenv("QDRANT_COLLECTION")
            or cfg.get("PERSIST_COLLECTION_TYPE")
            or cfg.get("QDRANT_COLLECTION")
            or "rag_llm"
        )
        ing_id = os.getenv("INGESTION_ID") or cfg.get("INGESTION_ID")
        p_dir = os.getenv("PERSIST_DIR") or cfg.get("PERSIST_DIR") or "persist"
        persist_art = (
            os.getenv("PERSIST_USER_ARTIFACT", "True").lower() in ("true", "1", "yes")
            if os.getenv("PERSIST_USER_ARTIFACT") is not None
            else bool(cfg.get("PERSIST_USER_ARTIFACT", True))
        )
        up_s3 = (
            os.getenv("UPLOAD_USER_ARTIFACT_TO_S3", "False").lower() in ("true", "1", "yes")
            if os.getenv("UPLOAD_USER_ARTIFACT_TO_S3") is not None
            else bool(cfg.get("UPLOAD_USER_ARTIFACT_TO_S3", False))
        )

        return cls(
            collection_type=col,
            ingestion_id=ing_id,
            persist_dir=p_dir,
            persist_artifact=persist_art,
            upload_to_s3=up_s3,
            chunking_config=ChunkingConfig.from_env(cfg),
            embedding_config=EmbeddingConfig.from_env(cfg),
            vector_config=VectorStoreConfig.from_env(cfg, collection_name=col),
            persistence_config=PersistenceConfig.from_env(cfg, collection_type=col),
            s3_config=S3UploadConfig.from_env(cfg, collection_type=col) if up_s3 else None,
        )


@dataclass
class UserPipelineResult:
    """Telemetry and outcome of a single user/request ingestion execution."""

    success: bool = False
    status: str = "PENDING"
    ingestion_id: str = ""
    collection_type: str = ""
    stage: str | None = None
    documents: int = 0
    chunks: int = 0
    embedded: int = 0
    failed_embeddings: int = 0
    persisted: bool = False
    validated: bool = False
    counts: StageCounts = field(default_factory=StageCounts)
    persistence_result: PersistenceResult | None = None
    vector_result: VectorStoreResult | None = None
    s3_result: S3UploadResult | None = None
    error: str | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize user pipeline result to JSON-serializable dictionary."""
        return {
            "success": self.success,
            "status": self.status,
            "ingestion_id": self.ingestion_id,
            "collection_type": self.collection_type,
            "stage": self.stage,
            "documents": self.documents,
            "chunks": self.chunks,
            "embedded": self.embedded,
            "failed_embeddings": self.failed_embeddings,
            "persisted": self.persisted,
            "validated": self.validated,
            "counts": {
                "files_seen": self.counts.files_seen,
                "docs_loaded": self.counts.docs_loaded,
                "docs_cleaned": self.counts.docs_cleaned,
                "docs_discarded": self.counts.docs_discarded,
                "chunks_created": self.counts.chunks_created,
                "embeddings_generated": self.counts.embeddings_generated,
                "chunks_failed": self.counts.chunks_failed,
                "vectors_inserted": self.counts.vectors_inserted,
                "vectors_failed": self.counts.vectors_failed,
            },
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 4),
        }


@dataclass
class ValidationResult:
    """Structured result of an authoritative validation pass across pipeline artifacts or stages."""

    valid: bool
    stage: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checks_performed: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def raise_if_invalid(self) -> None:
        """Raise ValidationError if validation failed."""
        if not self.valid:
            from errors import ValidationError

            err_msg = "; ".join(self.errors) if self.errors else "Unknown validation failure"
            raise ValidationError(stage=self.stage, reason=err_msg, check="integrity")

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result to dictionary."""
        return {
            "valid": self.valid,
            "stage": self.stage,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks_performed": self.checks_performed,
            "details": self.details,
        }



