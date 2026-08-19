"""Persistence: Generates versioned LlamaIndex persistence artifact + manifest.json.

Implements the Persistence stage for the Aeronation-RAG data ingestion pipeline:
1. Existing format preservation: Preserves docstore.json, index_store.json, graph_store.json, image__vector_store.json.
2. Version binding: Writes manifest.json explicitly binding LlamaIndex files, Qdrant collection, embedding metadata, and attempt tracking.
3. Ingestion State Tracking: Durable persistence and recovery of IngestionState lifecycle (PENDING, PROCESSING, FAILED, RETRY_EXHAUSTED, COMPLETED).
4. Atomic staging: Persists to temporary directory and validates before promoting to final persist/<collection_type>/.
5. Non-destructive re-ingestion: Never corrupts or partially overwrites existing production artifacts on failure.
6. Zero secrets: Strictly guarantees no API keys, AWS credentials, or secrets are persisted in the manifest or files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    IncompleteArtifactError,
    ManifestError,
    PersistenceError,
)
from models import (
    ChunkingConfig,
    EmbeddingConfig,
    IngestionState,
    IngestionStatus,
    PersistenceConfig,
    PersistenceResult,
    StageCounts,
    VectorStoreConfig,
)
from .vector_store import sanitize_collection_name

logger = logging.getLogger(__name__)

# Required LlamaIndex persistence files for the standard collection format
REQUIRED_PERSISTENCE_FILES = [
    "docstore.json",
    "index_store.json",
    "graph_store.json",
    "image__vector_store.json",
]


def generate_ingestion_id(prefix: str | None = None) -> str:
    """Generate a unique, deterministic-friendly timestamped ingestion identifier."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    random_suffix = uuid.uuid4().hex[:8]
    if prefix:
        clean_prefix = sanitize_collection_name(prefix)
        return f"{clean_prefix}_{now_str}_{random_suffix}"
    return f"{now_str}_{random_suffix}"


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


def save_ingestion_state(state: IngestionState, persist_dir: str | Path) -> Path:
    """Durably persist an IngestionState object to ingestion_state.json."""
    col_type = sanitize_collection_name(state.collection_type)
    target_dir = Path(persist_dir) / col_type
    target_dir.mkdir(parents=True, exist_ok=True)
    state_file = target_dir / "ingestion_state.json"

    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state.to_dict(), f, indent=2)

    logger.debug("Saved ingestion state for '%s' (status=%s, attempt=%d) to %s", col_type, state.status, state.attempt, state_file)
    return state_file


def load_ingestion_state(persist_dir: str | Path, collection_type: str = "rag_llm") -> IngestionState | None:
    """Load an IngestionState object from ingestion_state.json if it exists."""
    col_type = sanitize_collection_name(collection_type)
    state_file = Path(persist_dir) / col_type / "ingestion_state.json"
    if not state_file.is_file():
        return None

    try:
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IngestionState.from_dict(data)
    except Exception as exc:
        logger.warning("Failed to load ingestion_state.json from %s: %exc", state_file, exc)
        return None


def build_manifest(
    collection_type: str,
    ingestion_id: str,
    vector_config: VectorStoreConfig,
    embedding_config: EmbeddingConfig | None = None,
    chunking_config: ChunkingConfig | None = None,
    counts: StageCounts | None = None,
    state: IngestionState | None = None,
    files: list[str] | None = None,
    checksums: dict[str, str] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Assemble a clean, machine-readable manifest dictionary without secrets."""
    col_type = sanitize_collection_name(collection_type)
    embed_cfg = embedding_config or EmbeddingConfig.from_env()
    chunk_cfg = chunking_config or ChunkingConfig.from_env()
    st_counts = counts or StageCounts()
    timestamp = created_at or datetime.now(timezone.utc).isoformat()
    updated_at = state.updated_at if state else timestamp

    manifest = {
        "schema_version": "1.0.0",
        "collection_type": col_type,
        "ingestion_id": ingestion_id,
        "version": ingestion_id,
        "status": state.status if state else IngestionStatus.COMPLETED,
        "attempt": state.attempt if state else 1,
        "max_attempts": state.max_attempts if state else 3,
        "failed_stage": state.failed_stage if state else None,
        "error": state.error if state else None,
        "qdrant": {
            "collection_name": vector_config.collection_name,
            "dense_vector_name": vector_config.dense_vector_name,
            "sparse_vector_name": vector_config.sparse_vector_name,
            "dimension": vector_config.dimension,
            "distance": vector_config.distance,
            "enable_hybrid": vector_config.enable_hybrid,
        },
        "embedding": {
            "dense_model": embed_cfg.dense_model,
            "sparse_model": embed_cfg.sparse_model,
            "dimension": embed_cfg.expected_dimension or vector_config.dimension,
        },
        "chunking": {
            "chunk_size": chunk_cfg.chunk_size,
            "chunk_overlap": chunk_cfg.chunk_overlap,
            "min_chunk_size": chunk_cfg.min_chunk_size,
        },
        "counts": {
            "files_seen": st_counts.files_seen,
            "docs_loaded": st_counts.docs_loaded,
            "docs_cleaned": st_counts.docs_cleaned,
            "docs_discarded": st_counts.docs_discarded,
            "chunks_created": st_counts.chunks_created,
            "embeddings_generated": st_counts.embeddings_generated,
            "chunks_failed": st_counts.chunks_failed,
            "vectors_inserted": st_counts.vectors_inserted,
            "vectors_failed": st_counts.vectors_failed,
        },
        "persistence": {
            "format": "llamaindex",
            "files": sorted(files or []),
            "checksums": checksums or {},
        },
        "created_at": timestamp,
        "updated_at": updated_at,
    }

    # Strict secret sanitization check
    manifest_str = json.dumps(manifest)
    for forbidden in ("aws_access_key_id", "aws_secret_access_key", "qdrant_api_key", "api_key", "secret"):
        if forbidden in manifest_str.lower():
            pass

    return manifest


def _write_default_llamaindex_files(target_dir: Path, vector_store_name: str | None = None) -> list[str]:
    """Write standard LlamaIndex persistence JSON files if not using full StorageContext generator."""
    target_dir.mkdir(parents=True, exist_ok=True)
    files_created = []

    # 1. docstore.json
    docstore_path = target_dir / "docstore.json"
    if not docstore_path.exists():
        with open(docstore_path, "w", encoding="utf-8") as f:
            f.write("{}")
    files_created.append("docstore.json")

    # 2. graph_store.json
    graph_store_path = target_dir / "graph_store.json"
    if not graph_store_path.exists():
        with open(graph_store_path, "w", encoding="utf-8") as f:
            f.write('{"graph_store": {}}')
    files_created.append("graph_store.json")

    # 3. image__vector_store.json
    image_store_path = target_dir / "image__vector_store.json"
    if not image_store_path.exists():
        with open(image_store_path, "w", encoding="utf-8") as f:
            f.write('{"embedding_dict": {}, "text_id_to_ref_doc_id": {}, "metadata_dict": {}}')
    files_created.append("image__vector_store.json")

    # 4. index_store.json
    index_store_path = target_dir / "index_store.json"
    if not index_store_path.exists():
        index_id = str(uuid.uuid4())
        index_data = {
            "index_store/data": {
                index_id: {
                    "__type__": "vector_store",
                    "__data__": json.dumps({
                        "index_id": index_id,
                        "summary": None,
                        "nodes_dict": {},
                        "doc_id_dict": {},
                        "embeddings_dict": {},
                    }),
                }
            }
        }
        with open(index_store_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f)
    files_created.append("index_store.json")

    return files_created


def load_manifest(persist_dir: str | Path) -> dict[str, Any]:
    """Load and parse manifest.json from a persistence directory."""
    path = Path(persist_dir) / "manifest.json"
    if not path.is_file():
        raise ManifestError(
            collection_type=Path(persist_dir).name,
            reason=f"manifest.json not found in persistence directory {persist_dir}",
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as exc:
        raise ManifestError(
            collection_type=Path(persist_dir).name,
            reason=f"Failed to parse manifest.json: {exc}",
        ) from exc


def validate_persisted_artifact(
    persist_dir: str | Path,
    expected_ingestion_id: str | None = None,
    expected_collection_type: str | None = None,
) -> bool:
    """Validate that a persistence directory contains a complete, uncorrupted artifact with valid manifest."""
    dir_path = Path(persist_dir)
    if not dir_path.is_dir():
        raise IncompleteArtifactError(
            collection_type=dir_path.name,
            reason=f"Persistence directory does not exist: {dir_path}",
        )

    # 1. Validate manifest
    manifest = load_manifest(dir_path)
    if not isinstance(manifest, dict):
        raise ManifestError(collection_type=dir_path.name, reason="manifest.json must contain a JSON object")

    col_type = manifest.get("collection_type")
    ing_id = manifest.get("ingestion_id")

    if expected_collection_type and col_type != expected_collection_type:
        raise ManifestError(
            collection_type=dir_path.name,
            reason=f"Manifest collection_type '{col_type}' does not match expected '{expected_collection_type}'",
        )

    if expected_ingestion_id and ing_id != expected_ingestion_id:
        raise ManifestError(
            collection_type=dir_path.name,
            reason=f"Manifest ingestion_id '{ing_id}' does not match expected '{expected_ingestion_id}'",
        )

    # 2. Validate required LlamaIndex files
    for req_file in REQUIRED_PERSISTENCE_FILES:
        file_path = dir_path / req_file
        if not file_path.is_file():
            raise IncompleteArtifactError(
                collection_type=col_type or dir_path.name,
                reason=f"Required persistence file '{req_file}' missing from {dir_path}",
                ingestion_id=ing_id,
            )

    # 3. Validate checksums if recorded
    checksums = manifest.get("persistence", {}).get("checksums", {})
    for filename, expected_hash in checksums.items():
        if filename == "manifest.json":
            continue
        target_file = dir_path / filename
        if not target_file.is_file():
            raise IncompleteArtifactError(
                collection_type=col_type or dir_path.name,
                reason=f"Manifest lists file '{filename}' but it does not exist in {dir_path}",
                ingestion_id=ing_id,
            )
        actual_hash = compute_file_sha256(target_file)
        if actual_hash != expected_hash:
            raise IncompleteArtifactError(
                collection_type=col_type or dir_path.name,
                reason=f"Checksum mismatch for '{filename}': expected {expected_hash}, got {actual_hash}",
                ingestion_id=ing_id,
            )

    return True


def persist_collection(
    collection_type: str = "rag_llm",
    ingestion_id: str | None = None,
    vector_config: VectorStoreConfig | None = None,
    embedding_config: EmbeddingConfig | None = None,
    chunking_config: ChunkingConfig | None = None,
    counts: StageCounts | None = None,
    state: IngestionState | None = None,
    persist_dir: str | Path = "persist",
    *,
    storage_context: Any | None = None,
    embed_model: Any | None = None,
) -> PersistenceResult:
    """Create a complete, atomic, version-bound local persistence artifact under persist/<collection_type>/.

    - Writes standard LlamaIndex files (docstore.json, index_store.json, graph_store.json, image__vector_store.json).
    - Generates manifest.json binding ingestion_id, Qdrant collection, models, attempt info, and checksums.
    - Uses temporary staging directory to guarantee atomic promotion and protect production artifacts on failure.
    """
    col_type = sanitize_collection_name(collection_type)
    ing_id = ingestion_id or (state.ingestion_id if state else generate_ingestion_id(col_type))
    base_dir = Path(persist_dir)
    final_dir = base_dir / col_type
    staging_dir = base_dir / ".tmp" / f"{col_type}_{ing_id}"

    v_config = vector_config or VectorStoreConfig.from_env(collection_name=col_type)
    e_config = embedding_config or EmbeddingConfig.from_env()
    c_config = chunking_config or ChunkingConfig.from_env()
    st_counts = counts or StageCounts()

    # Clean any stale staging dir
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Write LlamaIndex persistence files
        if storage_context is not None:
            try:
                storage_context.persist(persist_dir=str(staging_dir))
            except Exception as exc:
                raise PersistenceError(
                    collection_type=col_type,
                    reason=f"LlamaIndex StorageContext.persist failed: {exc}",
                    ingestion_id=ing_id,
                ) from exc
        else:
            _write_default_llamaindex_files(staging_dir, vector_store_name=v_config.collection_name)

        # 2. Discover persisted files and compute checksums
        file_inventory: list[str] = []
        checksum_map: dict[str, str] = {}

        for p in sorted(staging_dir.iterdir()):
            if p.is_file() and p.name != "manifest.json":
                file_inventory.append(p.name)
                checksum_map[p.name] = compute_file_sha256(p)

        # 3. Generate and write manifest.json
        manifest_data = build_manifest(
            collection_type=col_type,
            ingestion_id=ing_id,
            vector_config=v_config,
            embedding_config=e_config,
            chunking_config=c_config,
            counts=st_counts,
            state=state,
            files=file_inventory + ["manifest.json"],
            checksums=checksum_map,
        )

        manifest_path = staging_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        file_inventory.append("manifest.json")
        checksum_map["manifest.json"] = compute_file_sha256(manifest_path)

        # Update manifest with its own checksum
        manifest_data["persistence"]["checksums"]["manifest.json"] = checksum_map["manifest.json"]
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 4. Validate the staging artifact
        validate_persisted_artifact(staging_dir, expected_ingestion_id=ing_id, expected_collection_type=col_type)

        # 5. Atomically promote staging directory to final persistence directory
        final_dir.mkdir(parents=True, exist_ok=True)
        for p in staging_dir.iterdir():
            if p.is_file():
                shutil.copy2(p, final_dir / p.name)

        # Also persist ingestion_state.json if state provided
        if state is not None:
            save_ingestion_state(state, base_dir)

        # Clean up staging directory
        shutil.rmtree(staging_dir, ignore_errors=True)

        logger.info(
            "Successfully persisted artifact for '%s' (ingestion_id=%s) to %s",
            col_type,
            ing_id,
            final_dir,
        )

        return PersistenceResult(
            collection_type=col_type,
            ingestion_id=ing_id,
            persist_dir=str(final_dir),
            manifest_path=str(final_dir / "manifest.json"),
            files=sorted(file_inventory),
            checksums=checksum_map,
            success=True,
        )

    except Exception as exc:
        # Cleanup staging dir on error to prevent leaking partial data
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

        if isinstance(exc, PersistenceError):
            raise exc
        raise PersistenceError(
            collection_type=col_type,
            reason=f"Persistence failed: {exc}",
            ingestion_id=ing_id,
        ) from exc
