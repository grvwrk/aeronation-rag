"""
Aeronation RAG Ingestion Pipeline Package.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility shim for newer qdrant-client versions with llama-index-vector-stores-qdrant
try:
    import qdrant_client.qdrant_fastembed as _qf
    if not hasattr(_qf, "IDF_EMBEDDING_MODELS"):
        _qf.IDF_EMBEDDING_MODELS = []
except Exception:
    pass

from .shared_processing import (
    EmbeddingEngine,
    build_manifest,
    chunk_document,
    chunk_documents,
    clean_document,
    clean_documents,
    clean_text,
    cleanup_staging_artifacts,
    compute_file_sha256,
    create_collection,
    derive_point_id,
    embed_chunk,
    embed_chunks,
    generate_chunk_id,
    generate_ingestion_id,
    get_collection_info,
    get_qdrant_client,
    load_directory,
    load_file,
    load_ingestion_state,
    load_manifest,
    load_urls,
    normalize_hyphenated_line_breaks,
    normalize_line_endings,
    normalize_whitespace,
    pack_chunks_with_overlap,
    persist_collection,
    remove_control_characters,
    run_system_pipeline,
    run_user_pipeline,
    sanitize_collection_name,
    save_ingestion_state,
    split_paragraphs,
    split_sentences,
    stable_source_id,
    upsert_embeddings,
    validate_collection,
    validate_persisted_artifact,
    build_s3_key,
    cleanup_staged_s3_artifact,
    get_s3_client,
    promote_staged_s3_artifact,
    upload_directory_to_s3,
    upload_file_to_s3,
    upload_persistence_to_s3,
    verify_s3_objects,
)

__all__ = [
    "clean_document",
    "clean_documents",
    "clean_text",
    "normalize_hyphenated_line_breaks",
    "normalize_line_endings",
    "normalize_whitespace",
    "remove_control_characters",
    "chunk_document",
    "chunk_documents",
    "generate_chunk_id",
    "pack_chunks_with_overlap",
    "split_paragraphs",
    "split_sentences",
    "embed_chunk",
    "embed_chunks",
    "EmbeddingEngine",
    "create_collection",
    "validate_collection",
    "upsert_embeddings",
    "get_collection_info",
    "sanitize_collection_name",
    "derive_point_id",
    "get_qdrant_client",
    "persist_collection",
    "load_manifest",
    "validate_persisted_artifact",
    "generate_ingestion_id",
    "build_manifest",
    "compute_file_sha256",
    "save_ingestion_state",
    "load_ingestion_state",
    "run_system_pipeline",
    "run_user_pipeline",
    "cleanup_staging_artifacts",
    "load_file",
    "load_directory",
    "load_urls",
    "stable_source_id",
    "upload_persistence_to_s3",
    "verify_s3_objects",
    "build_s3_key",
    "upload_directory_to_s3",
    "upload_file_to_s3",
    "promote_staged_s3_artifact",
    "cleanup_staged_s3_artifact",
    "get_s3_client",
]

