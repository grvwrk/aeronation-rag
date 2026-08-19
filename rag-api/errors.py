"""Shared error hierarchy for the ingestion pipeline (loader, cleaner, chunker, embedder, etc).
"""

from __future__ import annotations

from dataclasses import dataclass


class PipelineError(Exception):
    """Root of every error this package raises."""


class SourceError(PipelineError):
    """Base class for anything that goes wrong turning a source into RawDocs."""


@dataclass
class LoadFailure:
    """Structured description of a single load failure."""

    source: str
    reason: str
    page: int | None = None
    error_type: str = "LoadError"

    def __str__(self) -> str:
        lines = ["Loading failed:", f"  source = {self.source}"]
        if self.page is not None:
            lines.append(f"  page   = {self.page}")
        lines.append(f"  reason = {self.reason}")
        return "\n".join(lines)


class LoaderError(SourceError):
    """Base class for all loader-stage errors."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        page: int | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = LoadFailure(
            source=source,
            reason=reason,
            page=page,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


class SourceNotFoundError(LoaderError):
    """The source path doesn't exist, isn't a file, or the URL is unreachable."""


class SourcePermissionError(LoaderError):
    """The source exists but can't be read — filesystem permissions, or HTTP 401/403."""


class UnsupportedFormatError(LoaderError):
    """The source's extension isn't one the loader handles."""


class EmptySourceError(LoaderError):
    """The source is reachable and readable but contains no content."""


class ParseError(LoaderError):
    """The source was read successfully but couldn't be turned into text."""


@dataclass
class CleanFailure:
    """Structured description of a single clean failure."""

    source: str
    reason: str
    page: int | None = None
    error_type: str = "CleanError"

    def __str__(self) -> str:
        lines = ["Cleaning failed:", f"  source = {self.source}"]
        if self.page is not None:
            lines.append(f"  page   = {self.page}")
        lines.append(f"  reason = {self.reason}")
        return "\n".join(lines)


class CleanerError(PipelineError):
    """Base class for all cleaner-stage errors."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        page: int | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = CleanFailure(
            source=source,
            reason=reason,
            page=page,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


@dataclass
class ChunkFailure:
    """Structured description of a single chunk failure."""

    source: str
    reason: str
    page: int | None = None
    chunk_index: int | None = None
    error_type: str = "ChunkError"

    def __str__(self) -> str:
        lines = ["Chunking failed:", f"  source = {self.source}"]
        if self.page is not None:
            lines.append(f"  page   = {self.page}")
        if self.chunk_index is not None:
            lines.append(f"  chunk  = {self.chunk_index}")
        lines.append(f"  reason = {self.reason}")
        return "\n".join(lines)


class ChunkerError(PipelineError):
    """Base class for all chunker-stage errors."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        page: int | None = None,
        chunk_index: int | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = ChunkFailure(
            source=source,
            reason=reason,
            page=page,
            chunk_index=chunk_index,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


@dataclass
class EmbedFailure:
    """Structured description of a single embedding failure."""

    source: str
    reason: str
    chunk_id: str | None = None
    model_name: str | None = None
    error_type: str = "EmbedError"

    def __str__(self) -> str:
        lines = ["Embedding failed:", f"  source = {self.source}"]
        if self.chunk_id is not None:
            lines.append(f"  chunk_id = {self.chunk_id}")
        if self.model_name is not None:
            lines.append(f"  model    = {self.model_name}")
        lines.append(f"  reason   = {self.reason}")
        return "\n".join(lines)


class EmbedderError(PipelineError):
    """Base class for all embedder-stage errors."""

    def __init__(
        self,
        source: str,
        reason: str,
        *,
        chunk_id: str | None = None,
        model_name: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = EmbedFailure(
            source=source,
            reason=reason,
            chunk_id=chunk_id,
            model_name=model_name,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


class ModelLoadError(EmbedderError):
    """Raised when an embedding model fails to initialize, download, or load."""


class DimensionMismatchError(EmbedderError):
    """Raised when generated embedding vector dimension does not match expected dimension."""


@dataclass
class VectorStoreFailure:
    """Structured description of a single vector store failure."""

    collection: str
    reason: str
    source_id: str | None = None
    chunk_id: str | None = None
    error_type: str = "VectorStoreError"

    def __str__(self) -> str:
        lines = ["VectorStore operation failed:", f"  collection = {self.collection}"]
        if self.source_id is not None:
            lines.append(f"  source     = {self.source_id}")
        if self.chunk_id is not None:
            lines.append(f"  chunk_id   = {self.chunk_id}")
        lines.append(f"  reason     = {self.reason}")
        return "\n".join(lines)


class VectorStoreError(PipelineError):
    """Base class for all vector store stage errors."""

    def __init__(
        self,
        collection: str,
        reason: str,
        *,
        source_id: str | None = None,
        chunk_id: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = VectorStoreFailure(
            collection=collection,
            reason=reason,
            source_id=source_id,
            chunk_id=chunk_id,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


class IncompatibleCollectionError(VectorStoreError):
    """Raised when an existing Qdrant collection has incompatible dimension, metric, or vector config."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when a requested Qdrant collection does not exist."""


class VectorDimensionError(VectorStoreError):
    """Raised when an embedding vector dimension does not match collection configuration."""


@dataclass
class PersistenceFailure:
    """Structured description of a single persistence failure."""

    collection_type: str
    reason: str
    ingestion_id: str | None = None
    error_type: str = "PersistenceError"

    def __str__(self) -> str:
        lines = ["Persistence operation failed:", f"  collection_type = {self.collection_type}"]
        if self.ingestion_id is not None:
            lines.append(f"  ingestion_id    = {self.ingestion_id}")
        lines.append(f"  reason          = {self.reason}")
        return "\n".join(lines)


class PersistenceError(PipelineError):
    """Base class for all persistence stage errors."""

    def __init__(
        self,
        collection_type: str,
        reason: str,
        *,
        ingestion_id: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = PersistenceFailure(
            collection_type=collection_type,
            reason=reason,
            ingestion_id=ingestion_id,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


class ManifestError(PersistenceError):
    """Raised when manifest generation, serialization, or validation fails."""


class IncompleteArtifactError(PersistenceError):
    """Raised when required persistence files or checksums are missing or corrupt."""


class PipelineExecutionError(PipelineError):
    """Base class for pipeline execution and orchestration errors."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        ingestion_id: str | None = None,
        attempt: int | None = None,
    ) -> None:
        self.stage = stage
        self.ingestion_id = ingestion_id
        self.attempt = attempt
        super().__init__(message)


class StageExecutionError(PipelineExecutionError):
    """Raised when a specific pipeline stage fails execution."""


class RetryExhaustedError(PipelineExecutionError):
    """Raised when the maximum number of re-ingestion attempts is exhausted."""


@dataclass
class S3UploadFailure:
    """Structured description of an S3 operation failure."""

    bucket: str
    prefix: str
    reason: str
    key: str | None = None
    error_type: str = "S3Error"

    def __str__(self) -> str:
        lines = [f"{self.error_type}:", f"  bucket = {self.bucket}", f"  prefix = {self.prefix}"]
        if self.key is not None:
            lines.append(f"  key    = {self.key}")
        lines.append(f"  reason = {self.reason}")
        return "\n".join(lines)


class S3Error(PipelineError):
    """Base class for all S3 upload, validation, and storage errors."""

    def __init__(
        self,
        bucket: str,
        prefix: str,
        reason: str,
        *,
        key: str | None = None,
        error_type: str | None = None,
    ) -> None:
        self.failure = S3UploadFailure(
            bucket=bucket,
            prefix=prefix,
            reason=reason,
            key=key,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


class S3ConfigurationError(S3Error):
    """Raised when S3 bucket, prefix, region, or credentials configuration is missing or invalid."""


class S3UploadError(S3Error):
    """Raised when uploading persistence files to S3 fails (partial or complete failure)."""


class S3VerificationError(S3Error):
    """Raised when verified S3 objects, counts, or manifest do not match expected local artifacts."""


class S3PermissionError(S3Error):
    """Raised when AWS S3 returns AccessDenied (HTTP 403) or insufficient IAM permissions."""


class S3ObjectNotFoundError(S3Error):
    """Raised when a required S3 object or manifest key is not found in the bucket."""


@dataclass
class ValidationFailure:
    """Structured description of a validation check failure."""

    stage: str
    check: str
    reason: str
    error_type: str = "ValidationError"

    def __str__(self) -> str:
        return f"Validation failed:\n  stage  = {self.stage}\n  check  = {self.check}\n  reason = {self.reason}"


class ValidationError(PipelineError):
    """Base class for validation failures across all pipeline stages."""

    def __init__(
        self,
        stage: str,
        reason: str,
        *,
        check: str = "general",
        error_type: str | None = None,
    ) -> None:
        self.failure = ValidationFailure(
            stage=stage,
            check=check,
            reason=reason,
            error_type=error_type or type(self).__name__,
        )
        super().__init__(str(self.failure))


