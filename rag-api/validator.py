"""
Authoritative Validation Layer for the Aeronation RAG Ingestion Pipeline.

Responsibilities:
1. Configuration validation: Pre-flight checks for chunking, embedding, vector store, retries, and storage.
2. Collection type validation: Strict sanitization and path traversal prevention.
3. Raw & Cleaned Document validation: Schema structure, non-empty content, metadata integrity.
4. Chunk validation: Deterministic IDs, non-empty chunks, token boundary checks.
5. Embedding validation: Dense dimensions, sparse indices/values consistency, hybrid pairing, zero silent loss.
6. Vector store validation: Read-only verification of collection existence, dimension, and distance metrics.
7. Persistence & Manifest validation: File completeness, SHA-256 checksums, version binding, and loadability.
8. Stage Count consistency: Logical count invariants across pipeline stages.
9. S3 Artifact validation: Remote object completeness and version matching.
10. High-level Ingestion validation: Unified multi-stage verification before promotion.

STRICT READ-ONLY:
- Does NOT load or parse original source documents.
- Does NOT clean, chunk, or embed text.
- Does NOT create or modify Qdrant collections or points.
- Does NOT write or alter local persistence files.
- Does NOT upload or mutate S3 objects.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import hashlib

from errors import (
    IncompleteArtifactError,
    ManifestError,
    ValidationError,
)

# Standard LlamaIndex storage persistence files
REQUIRED_PERSISTENCE_FILES: tuple[str, ...] = (
    "docstore.json",
    "index_store.json",
    "graph_store.json",
    "image__vector_store.json",
    "manifest.json",
)


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute standard SHA-256 hash of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute checksum: file not found at {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def load_manifest(persist_dir: Path | str) -> dict[str, Any]:
    """Load and parse manifest.json from a persistence directory."""
    manifest_path = Path(persist_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise ManifestError(
            collection_type=Path(persist_dir).name,
            reason=f"manifest.json does not exist in {persist_dir}",
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise ManifestError(
            collection_type=Path(persist_dir).name,
            reason=f"Corrupted or invalid JSON in manifest.json: {exc}",
        ) from exc

from models import (
    Chunk,
    ChunkingConfig,
    EmbeddedChunk,
    EmbeddingConfig,
    FailedChunk,
    PipelineConfig,
    RawDoc,
    S3UploadConfig,
    StageCounts,
    UserPipelineConfig,
    ValidationResult,
    VectorStoreConfig,
)

logger = logging.getLogger(__name__)

# Strict collection name regex: alphanumeric, underscores, hyphens only.
_COLLECTION_TYPE_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


# --------------------------------------------------------------------------- #
# 1. Collection Type Validation
# --------------------------------------------------------------------------- #


def validate_collection_type(collection_type: str | Any) -> str:
    """Validate and sanitize a collection type name.

    Guarantees:
    - Must be a non-empty string.
    - Rejects path traversal attempts (e.g. '../rag_llm', 'a/../../b').
    - Rejects absolute and relative filesystem paths (containing '/' or '\\').
    - Only allows safe characters: [a-zA-Z0-9_-].

    Raises:
        ValueError: If collection_type is invalid, empty, or contains unsafe characters.

    Returns:
        str: Cleaned collection type string.
    """
    if not isinstance(collection_type, str):
        raise ValueError(
            f"Collection type must be a string, got {type(collection_type).__name__}"
        )

    trimmed = collection_type.strip()
    if not trimmed:
        raise ValueError("Collection type cannot be empty or whitespace-only")

    # Defensive path traversal checks
    if ".." in trimmed or "/" in trimmed or "\\" in trimmed:
        raise ValueError(
            f"Invalid collection type '{collection_type}'. Path traversal and directory separators are strictly prohibited."
        )

    if not _COLLECTION_TYPE_RE.match(trimmed):
        raise ValueError(
            f"Invalid collection type '{trimmed}'. Only alphanumeric characters, underscores, and hyphens are allowed."
        )

    return trimmed


# --------------------------------------------------------------------------- #
# 2. Configuration Validation
# --------------------------------------------------------------------------- #


def validate_config(
    config: PipelineConfig | UserPipelineConfig | S3UploadConfig | dict[str, Any],
    *,
    check_aws: bool = False,
    check_qdrant: bool = False,
) -> ValidationResult:
    """Validate pipeline configuration parameters before ingestion execution."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 0

    # Extract configs
    if isinstance(config, (PipelineConfig, UserPipelineConfig)):
        col_type = config.collection_type
        chunk_cfg = config.chunking_config
        emb_cfg = config.embedding_config
        vec_cfg = config.vector_config
    elif isinstance(config, S3UploadConfig):
        col_type = config.collection_type
        chunk_cfg = None
        emb_cfg = None
        vec_cfg = None
    elif isinstance(config, dict):
        col_type = config.get("collection_type") or config.get("PERSIST_COLLECTION_TYPE")
        chunk_cfg = ChunkingConfig.from_env(config)
        emb_cfg = EmbeddingConfig.from_env(config)
        vec_cfg = VectorStoreConfig.from_env(config)
    else:
        errors.append(f"Unsupported config type: {type(config).__name__}")
        return ValidationResult(valid=False, stage="config", errors=errors, checks_performed=1)

    # 1. Collection type check
    checks += 1
    if col_type:
        try:
            validate_collection_type(col_type)
        except ValueError as exc:
            errors.append(f"Invalid collection_type: {exc}")
    else:
        errors.append("Collection type is missing or not configured")

    # 2. Chunking configuration check
    if chunk_cfg:
        checks += 3
        if chunk_cfg.chunk_size <= 0:
            errors.append(f"chunk_size must be positive, got {chunk_cfg.chunk_size}")
        if chunk_cfg.chunk_overlap < 0:
            errors.append(f"chunk_overlap cannot be negative, got {chunk_cfg.chunk_overlap}")
        if chunk_cfg.chunk_overlap >= chunk_cfg.chunk_size:
            errors.append(
                f"chunk_overlap ({chunk_cfg.chunk_overlap}) must be less than chunk_size ({chunk_cfg.chunk_size})"
            )

    # 3. Embedding configuration check
    if emb_cfg:
        checks += 2
        if not emb_cfg.dense_model or not str(emb_cfg.dense_model).strip():
            errors.append("Dense embedding model name is empty or missing")
        if emb_cfg.expected_dimension is not None and emb_cfg.expected_dimension <= 0:
            errors.append(f"expected_dimension must be a positive integer, got {emb_cfg.expected_dimension}")

    # 4. Vector store configuration check
    if vec_cfg:
        checks += 3
        if vec_cfg.dimension <= 0:
            errors.append(f"Vector dimension must be positive, got {vec_cfg.dimension}")
        if vec_cfg.distance not in ("Cosine", "Euclid", "Dot", "Manhattan"):
            errors.append(f"Invalid vector distance metric '{vec_cfg.distance}'")
        if vec_cfg.enable_hybrid and not vec_cfg.sparse_vector_name:
            errors.append("Hybrid vector store enabled but sparse_vector_name is missing")

        if check_qdrant:
            checks += 1
            if not vec_cfg.url:
                errors.append("Remote Qdrant check requested but QDRANT_URL is not configured")

    # 5. AWS / S3 configuration check
    if check_aws:
        checks += 1
        s3_cfg = getattr(config, "s3_config", None)
        if isinstance(config, S3UploadConfig):
            s3_cfg = config
        if s3_cfg and not s3_cfg.bucket_name:
            errors.append("S3 bucket name is missing or not configured")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="config",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"collection_type": col_type},
    )


# --------------------------------------------------------------------------- #
# 3. Raw Document Validation
# --------------------------------------------------------------------------- #


def validate_documents(
    docs: list[RawDoc] | Iterable[RawDoc],
    *,
    allow_empty: bool = False,
) -> ValidationResult:
    """Validate loaded raw document structures and content."""
    doc_list = list(docs)
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    if not doc_list:
        if not allow_empty:
            errors.append("Document list is empty (zero documents loaded)")
        return ValidationResult(
            valid=len(errors) == 0,
            stage="loader",
            errors=errors,
            warnings=warnings,
            checks_performed=checks,
            details={"documents_received": 0},
        )

    valid_docs = 0
    invalid_docs = 0

    for idx, doc in enumerate(doc_list):
        checks += 1
        if not isinstance(doc, RawDoc):
            errors.append(f"Document at index {idx} is not an instance of RawDoc (type: {type(doc).__name__})")
            invalid_docs += 1
            continue

        if not isinstance(doc.text, str) or not doc.text.strip():
            errors.append(f"Document at index {idx} (source_id: '{getattr(doc, 'source_id', 'unknown')}') contains empty or invalid text")
            invalid_docs += 1
            continue

        if not getattr(doc, "source_id", None) or not str(doc.source_id).strip():
            errors.append(f"Document at index {idx} has missing or empty source_id")
            invalid_docs += 1
            continue

        if not isinstance(getattr(doc, "metadata", {}), dict):
            errors.append(f"Document at index {idx} metadata must be a dictionary, got {type(doc.metadata).__name__}")
            invalid_docs += 1
            continue

        valid_docs += 1

    return ValidationResult(
        valid=len(errors) == 0,
        stage="loader",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={
            "documents_received": len(doc_list),
            "valid_documents": valid_docs,
            "invalid_documents": invalid_docs,
        },
    )


# --------------------------------------------------------------------------- #
# 4. Cleaned Document Validation
# --------------------------------------------------------------------------- #


def validate_cleaned_documents(
    cleaned_docs: list[RawDoc],
    *,
    original_docs: list[RawDoc] | None = None,
    allow_empty: bool = False,
) -> ValidationResult:
    """Validate cleaned document artifacts and verify document preservation."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    if original_docs and len(original_docs) > 0 and len(cleaned_docs) == 0 and not allow_empty:
        errors.append(
            f"Cleaner discarded all documents (received {len(original_docs)} input documents, produced 0 cleaned)"
        )
        return ValidationResult(
            valid=False,
            stage="cleaner",
            errors=errors,
            warnings=warnings,
            checks_performed=checks,
            details={"original_count": len(original_docs), "cleaned_count": 0},
        )

    # Validate each cleaned document
    doc_res = validate_documents(cleaned_docs, allow_empty=allow_empty)
    errors.extend(doc_res.errors)
    warnings.extend(doc_res.warnings)
    checks += doc_res.checks_performed

    return ValidationResult(
        valid=len(errors) == 0,
        stage="cleaner",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={
            "original_count": len(original_docs) if original_docs else len(cleaned_docs),
            "cleaned_count": len(cleaned_docs),
        },
    )


# --------------------------------------------------------------------------- #
# 5. Chunk Validation
# --------------------------------------------------------------------------- #


def validate_chunks(
    chunks: list[Chunk],
    *,
    config: ChunkingConfig | None = None,
    max_token_limit: int | None = None,
) -> ValidationResult:
    """Validate chunks produced by the chunker stage."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    if not chunks:
        errors.append("Chunk list is empty (0 chunks produced)")
        return ValidationResult(valid=False, stage="chunker", errors=errors, checks_performed=checks)

    seen_chunk_ids: set[str] = set()
    token_budget = max_token_limit or (config.chunk_size if config else None)

    for idx, ch in enumerate(chunks):
        checks += 1
        if not isinstance(ch, Chunk):
            errors.append(f"Item at index {idx} is not a Chunk instance (type: {type(ch).__name__})")
            continue

        if not ch.chunk_id or not str(ch.chunk_id).strip():
            errors.append(f"Chunk at index {idx} is missing a chunk_id")
        elif ch.chunk_id in seen_chunk_ids:
            errors.append(f"Duplicate chunk_id '{ch.chunk_id}' detected at index {idx}")
        else:
            seen_chunk_ids.add(ch.chunk_id)

        if not ch.text or not str(ch.text).strip():
            errors.append(f"Chunk '{ch.chunk_id}' contains empty text")

        if not ch.source_id or not str(ch.source_id).strip():
            errors.append(f"Chunk '{ch.chunk_id}' has missing or empty source_id")

        # Check token budget if recorded
        if token_budget and isinstance(ch.metadata, dict):
            tokens = ch.metadata.get("chunk_tokens")
            if isinstance(tokens, int) and tokens > token_budget * 1.5:  # Tolerance buffer for char-to-token variance
                errors.append(
                    f"Chunk '{ch.chunk_id}' token count ({tokens}) exceeds budget ({token_budget})"
                )

    return ValidationResult(
        valid=len(errors) == 0,
        stage="chunker",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"chunks_count": len(chunks), "unique_ids": len(seen_chunk_ids)},
    )


# --------------------------------------------------------------------------- #
# 6. Embedding Validation (Zero Silent Loss)
# --------------------------------------------------------------------------- #


def validate_embeddings(
    chunks: list[Chunk],
    embedded_chunks: list[EmbeddedChunk],
    *,
    failed_chunks: list[FailedChunk] | None = None,
    config: EmbeddingConfig | None = None,
    require_hybrid: bool = False,
) -> ValidationResult:
    """Strictly validate generated embeddings against input chunks."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    expected_dim = config.expected_dimension if config else None
    failed_list = failed_chunks or []

    # 1. Zero Silent Loss check: All valid chunks must be embedded successfully
    if len(chunks) != len(embedded_chunks):
        errors.append(
            f"Embedding count mismatch: received {len(chunks)} chunks, successfully embedded {len(embedded_chunks)}"
        )

    if failed_list:
        errors.append(f"{len(failed_list)} chunks failed embedding generation")

    expected_chunk_ids = {ch.chunk_id for ch in chunks}
    embedded_chunk_ids: set[str] = set()

    # 2. Vector dimension and representation check
    for idx, ech in enumerate(embedded_chunks):
        checks += 1
        if not isinstance(ech, EmbeddedChunk):
            errors.append(f"Item at index {idx} is not an EmbeddedChunk instance")
            continue

        cid = ech.chunk.chunk_id
        embedded_chunk_ids.add(cid)

        # Dense embedding check
        dense = ech.dense_embedding
        if not dense or not isinstance(dense, list):
            errors.append(f"Embedded chunk '{cid}' has missing or non-list dense embedding")
        else:
            if expected_dim is not None and len(dense) != expected_dim:
                errors.append(
                    f"Dense vector dimension mismatch for chunk '{cid}': expected {expected_dim}, got {len(dense)}"
                )
            if not all(isinstance(v, (int, float)) and not math.isnan(v) for v in dense):
                errors.append(f"Dense vector for chunk '{cid}' contains non-numeric or NaN values")

        # Sparse embedding check
        if require_hybrid or ech.sparse_embedding is not None:
            sparse = ech.sparse_embedding
            if sparse is None:
                if require_hybrid:
                    errors.append(f"Hybrid retrieval required but chunk '{cid}' has no sparse embedding")
            else:
                indices = getattr(sparse, "indices", None)
                values = getattr(sparse, "values", None)
                if indices is None or values is None:
                    errors.append(f"Sparse embedding for chunk '{cid}' is missing indices or values")
                elif len(indices) != len(values):
                    errors.append(
                        f"Sparse embedding for chunk '{cid}' length mismatch: {len(indices)} indices vs {len(values)} values"
                    )
                elif not all(isinstance(i, int) for i in indices):
                    errors.append(f"Sparse indices for chunk '{cid}' must be integers")
                elif not all(isinstance(v, (int, float)) and not math.isnan(v) for v in values):
                    errors.append(f"Sparse values for chunk '{cid}' must be valid numeric values")

    # 3. Completeness check
    missing_ids = expected_chunk_ids - embedded_chunk_ids
    if missing_ids:
        errors.append(f"Missing embeddings for {len(missing_ids)} chunk IDs: {list(missing_ids)[:5]}")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="embedder",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={
            "chunks_expected": len(chunks),
            "chunks_embedded": len(embedded_chunks),
            "chunks_failed": len(failed_list),
        },
    )


# --------------------------------------------------------------------------- #
# 7. Vector Store Validation (Read-Only)
# --------------------------------------------------------------------------- #


def validate_vector_store(
    client: Any,
    config: VectorStoreConfig,
    *,
    expected_points: int | None = None,
) -> ValidationResult:
    """Read-only validation of vector store collection existence and schema compatibility."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    col_name = config.collection_name
    try:
        # Check collection existence
        if hasattr(client, "collection_exists"):
            exists = client.collection_exists(col_name)
        else:
            exists = False
            try:
                client.get_collection(col_name)
                exists = True
            except Exception:
                exists = False

        if not exists:
            errors.append(f"Qdrant collection '{col_name}' does not exist")
            return ValidationResult(valid=False, stage="vector_store", errors=errors, checks_performed=checks)

        # Inspect collection config
        checks += 1
        col_info = client.get_collection(col_name)
        params = getattr(col_info, "config", None)
        params = getattr(params, "params", None) if params else None
        vectors_config = getattr(params, "vectors", None) if params else None

        if vectors_config and isinstance(vectors_config, dict):
            dense_cfg = vectors_config.get(config.dense_vector_name)
            if dense_cfg:
                dim = getattr(dense_cfg, "size", None)
                if dim is not None and dim != config.dimension:
                    errors.append(
                        f"Qdrant vector dimension mismatch: collection has {dim}, config expected {config.dimension}"
                    )
                dist = getattr(dense_cfg, "distance", None)
                dist_val = getattr(dist, "value", str(dist)) if dist else None
                if dist_val and dist_val.lower() != config.distance.lower():
                    errors.append(
                        f"Qdrant distance metric mismatch: collection has '{dist_val}', config expected '{config.distance}'"
                    )
            else:
                errors.append(f"Dense vector name '{config.dense_vector_name}' missing from collection config")

            if config.enable_hybrid and config.sparse_vector_name:
                checks += 1
                sparse_cfg = getattr(params, "sparse_vectors", None) if params else None
                if sparse_cfg and isinstance(sparse_cfg, dict) and config.sparse_vector_name not in sparse_cfg:
                    warnings.append(
                        f"Sparse vector '{config.sparse_vector_name}' not listed in collection sparse_vectors"
                    )

    except Exception as exc:
        errors.append(f"Failed inspecting Qdrant collection '{col_name}': {exc}")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="vector_store",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"collection_name": col_name},
    )


# --------------------------------------------------------------------------- #
# 8. Persistence & Manifest Validation
# --------------------------------------------------------------------------- #


def validate_manifest(
    manifest: dict[str, Any],
    *,
    expected_collection_type: str | None = None,
    expected_ingestion_id: str | None = None,
) -> ValidationResult:
    """Validate manifest.json schema, version binding, and file checksum table."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    if not isinstance(manifest, dict):
        errors.append("Manifest must be a dictionary")
        return ValidationResult(valid=False, stage="manifest", errors=errors, checks_performed=checks)

    # Required top-level keys
    for req_key in ("schema_version", "collection_type", "ingestion_id", "persistence"):
        checks += 1
        if req_key not in manifest:
            errors.append(f"Manifest missing required key: '{req_key}'")

    col_type = manifest.get("collection_type")
    ing_id = manifest.get("ingestion_id") or manifest.get("version")

    if expected_collection_type and col_type != expected_collection_type:
        errors.append(
            f"Manifest collection_type '{col_type}' does not match expected '{expected_collection_type}'"
        )

    if expected_ingestion_id and ing_id != expected_ingestion_id:
        errors.append(
            f"Manifest ingestion_id '{ing_id}' does not match expected '{expected_ingestion_id}'"
        )

    # Validate persistence block
    checks += 1
    p_info = manifest.get("persistence", {})
    if not isinstance(p_info, dict) or "files" not in p_info or "checksums" not in p_info:
        errors.append("Manifest persistence block must contain 'files' and 'checksums'")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="manifest",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"collection_type": col_type, "ingestion_id": ing_id},
    )


def validate_persistence(
    persist_dir: Path | str,
    *,
    expected_collection_type: str | None = None,
    expected_ingestion_id: str | None = None,
    verify_load: bool = False,
    vector_store: Any | None = None,
) -> ValidationResult:
    """Validate local persistence directory, files, checksums, and optional roundtrip loading."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    dir_path = Path(persist_dir)
    if not dir_path.exists() or not dir_path.is_dir():
        errors.append(f"Persistence directory does not exist or is not a directory: {dir_path}")
        return ValidationResult(valid=False, stage="persistence", errors=errors, checks_performed=checks)

    # 1. Validate manifest
    manifest_file = dir_path / "manifest.json"
    if not manifest_file.is_file():
        errors.append(f"manifest.json missing from persistence directory {dir_path}")
        return ValidationResult(valid=False, stage="persistence", errors=errors, checks_performed=checks)

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"manifest.json is not valid JSON: {exc}")
        return ValidationResult(valid=False, stage="persistence", errors=errors, checks_performed=checks)

    m_res = validate_manifest(
        manifest,
        expected_collection_type=expected_collection_type,
        expected_ingestion_id=expected_ingestion_id,
    )
    errors.extend(m_res.errors)
    checks += m_res.checks_performed

    col_type = manifest.get("collection_type", dir_path.name)
    ing_id = manifest.get("ingestion_id") or manifest.get("version")

    # 2. Validate all standard LlamaIndex persistence files exist
    for req_file in REQUIRED_PERSISTENCE_FILES:
        checks += 1
        fp = dir_path / req_file
        if not fp.is_file():
            errors.append(f"Required persistence file '{req_file}' missing from {dir_path}")

    # 3. Validate checksums recorded in manifest
    checksums = manifest.get("persistence", {}).get("checksums", {})
    for filename, expected_hash in checksums.items():
        if filename == "manifest.json":
            continue
        checks += 1
        fp = dir_path / filename
        if not fp.is_file():
            errors.append(f"Manifest lists file '{filename}' but it does not exist on disk")
            continue
        try:
            actual_hash = compute_file_sha256(fp)
            if actual_hash != expected_hash:
                errors.append(
                    f"Checksum mismatch for '{filename}': expected {expected_hash}, got {actual_hash}"
                )
        except Exception as exc:
            errors.append(f"Failed computing checksum for '{filename}': {exc}")

    # 4. Optional roundtrip loading verification
    if verify_load and len(errors) == 0:
        checks += 1
        try:
            from llama_index.core import StorageContext

            ctx = StorageContext.from_defaults(persist_dir=str(dir_path), vector_store=vector_store)
            if ctx.docstore is None or ctx.index_store is None:
                errors.append("StorageContext failed to load docstore or index_store from persistence")
        except Exception as exc:
            errors.append(f"StorageContext round-trip load failed: {exc}")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="persistence",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"collection_type": col_type, "ingestion_id": ing_id, "persist_dir": str(dir_path)},
    )


# --------------------------------------------------------------------------- #
# 9. Count Consistency Validation
# --------------------------------------------------------------------------- #


def validate_counts(
    counts: StageCounts,
    *,
    strict_embedding: bool = True,
) -> ValidationResult:
    """Validate logical relationships and stage count invariants."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 4

    # 1. Cleaned docs <= loaded docs
    if counts.docs_cleaned > counts.docs_loaded:
        errors.append(
            f"docs_cleaned ({counts.docs_cleaned}) cannot exceed docs_loaded ({counts.docs_loaded})"
        )

    # 2. Chunks created > 0 when cleaned docs > 0
    if counts.docs_cleaned > 0 and counts.chunks_created == 0:
        errors.append(f"0 chunks created despite {counts.docs_cleaned} cleaned documents")

    # 3. Strict embedding checks
    if strict_embedding:
        if counts.chunks_created > 0 and counts.embeddings_generated != counts.chunks_created:
            errors.append(
                f"Embedding count mismatch: {counts.chunks_created} chunks vs {counts.embeddings_generated} embeddings"
            )
        if counts.chunks_failed > 0:
            errors.append(f"{counts.chunks_failed} chunks recorded as failed during embedding")

    # 4. Vectors inserted == embeddings generated
    if counts.vectors_inserted != counts.embeddings_generated:
        errors.append(
            f"Vector store insert count mismatch: {counts.embeddings_generated} embeddings vs {counts.vectors_inserted} vectors inserted"
        )
    if counts.vectors_failed > 0:
        errors.append(f"{counts.vectors_failed} vectors failed during vector store upsert")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="counts",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={
            "docs_loaded": counts.docs_loaded,
            "chunks_created": counts.chunks_created,
            "embeddings_generated": counts.embeddings_generated,
            "vectors_inserted": counts.vectors_inserted,
        },
    )


# --------------------------------------------------------------------------- #
# 10. S3 Artifact Validation (Read-Only)
# --------------------------------------------------------------------------- #


def validate_s3_artifact(
    s3_client: Any,
    bucket: str,
    prefix: str,
    *,
    expected_collection_type: str | None = None,
    expected_ingestion_id: str | None = None,
    expected_files: list[str] | None = None,
) -> ValidationResult:
    """Read-only validation of remote S3 persistence artifacts and manifest."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    clean_prefix = prefix.strip("/")
    manifest_key = f"{clean_prefix}/manifest.json"

    # 1. Check remote manifest
    try:
        resp = s3_client.get_object(Bucket=bucket, Key=manifest_key)
        body = resp["Body"].read()
        manifest = json.loads(body.decode("utf-8"))
    except Exception as exc:
        errors.append(f"Failed retrieving/parsing remote manifest s3://{bucket}/{manifest_key}: {exc}")
        return ValidationResult(valid=False, stage="s3_validator", errors=errors, checks_performed=checks)

    m_res = validate_manifest(
        manifest,
        expected_collection_type=expected_collection_type,
        expected_ingestion_id=expected_ingestion_id,
    )
    errors.extend(m_res.errors)
    checks += m_res.checks_performed

    # 2. Check remote objects existence
    file_list = expected_files or REQUIRED_PERSISTENCE_FILES
    for req_file in file_list:
        checks += 1
        s3_key = f"{clean_prefix}/{req_file}"
        try:
            s3_client.head_object(Bucket=bucket, Key=s3_key)
        except Exception as exc:
            errors.append(f"Missing S3 object s3://{bucket}/{s3_key}: {exc}")

    return ValidationResult(
        valid=len(errors) == 0,
        stage="s3_validator",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"bucket": bucket, "prefix": clean_prefix},
    )


# --------------------------------------------------------------------------- #
# 11. High-Level Ingestion Result Validator
# --------------------------------------------------------------------------- #


def validate_ingestion_result(
    pipeline_result: Any,
    *,
    persist_dir: Path | str | None = None,
    client: Any | None = None,
    verify_load: bool = False,
) -> ValidationResult:
    """Multi-stage validation of complete pipeline output before promotion."""
    errors: list[str] = []
    warnings: list[str] = []
    checks = 1

    if not getattr(pipeline_result, "success", False):
        errors.append(
            f"Pipeline result is marked as failed (status='{getattr(pipeline_result, 'status', 'UNKNOWN')}', error='{getattr(pipeline_result, 'error', None)}')"
        )
        return ValidationResult(valid=False, stage="pipeline_result", errors=errors, checks_performed=checks)

    col_type = getattr(pipeline_result, "collection_type", None)
    ing_id = getattr(pipeline_result, "ingestion_id", None)
    counts = getattr(pipeline_result, "counts", None)

    # 1. Validate counts
    if counts:
        c_res = validate_counts(counts)
        errors.extend(c_res.errors)
        warnings.extend(c_res.warnings)
        checks += c_res.checks_performed

    # 2. Validate persistence
    target_persist = persist_dir or getattr(getattr(pipeline_result, "persistence_result", None), "persist_dir", None)
    if target_persist:
        p_res = validate_persistence(
            target_persist,
            expected_collection_type=col_type,
            expected_ingestion_id=ing_id,
            verify_load=verify_load,
        )
        errors.extend(p_res.errors)
        warnings.extend(p_res.warnings)
        checks += p_res.checks_performed

    return ValidationResult(
        valid=len(errors) == 0,
        stage="ingestion_validator",
        errors=errors,
        warnings=warnings,
        checks_performed=checks,
        details={"collection_type": col_type, "ingestion_id": ing_id},
    )


# --------------------------------------------------------------------------- #
# 12. Backward Compatibility Shim
# --------------------------------------------------------------------------- #


def validate_local_artifact(
    persist_dir: Path | str,
    *,
    expected_collection_type: str | None = None,
    expected_ingestion_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible validation function raising IncompleteArtifactError/ManifestError."""
    res = validate_persistence(
        persist_dir,
        expected_collection_type=expected_collection_type,
        expected_ingestion_id=expected_ingestion_id,
    )

    if not res.valid:
        col = res.details.get("collection_type", Path(persist_dir).name)
        ing = res.details.get("ingestion_id")
        first_err = res.errors[0] if res.errors else "Unknown artifact validation error"

        if "manifest" in first_err.lower():
            raise ManifestError(collection_type=col, reason=first_err)
        raise IncompleteArtifactError(collection_type=col, reason=first_err, ingestion_id=ing)

    return load_manifest(persist_dir)


def validate_persisted_artifact(
    persist_dir: Path | str,
    expected_ingestion_id: str | None = None,
    expected_collection_type: str | None = None,
) -> bool:
    """Validate persistence directory and return True on success, raising on failure."""
    validate_local_artifact(
        persist_dir,
        expected_collection_type=expected_collection_type,
        expected_ingestion_id=expected_ingestion_id,
    )
    return True

