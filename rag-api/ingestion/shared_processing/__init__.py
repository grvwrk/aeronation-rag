"""
Aeronation RAG Ingestion Pipeline Package.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility shim for newer qdrant-client versions with llama-index-vector-stores-qdrant
try:
    import qdrant_client.qdrant_fastembed as _qf
    if not hasattr(_qf, "IDF_EMBEDDING_MODELS"):
        _qf.IDF_EMBEDDING_MODELS = []
except Exception:
    pass

from .chunker import (
    chunk_document,
    chunk_documents,
    generate_chunk_id,
    pack_chunks_with_overlap,
    split_paragraphs,
    split_sentences,
)
from .cleaner import (
    clean_document,
    clean_documents,
    clean_text,
    normalize_hyphenated_line_breaks,
    normalize_line_endings,
    normalize_whitespace,
    remove_control_characters,
)
from .embedder import (
    EmbeddingEngine,
    embed_chunk,
    embed_chunks,
)
from .persistence import (
    build_manifest,
    compute_file_sha256,
    generate_ingestion_id,
    load_ingestion_state,
    load_manifest,
    persist_collection,
    save_ingestion_state,
    validate_persisted_artifact,
)
from .system_pipeline import (
    cleanup_staging_artifacts,
    run_system_pipeline,
)
from .user_pipeline import (
    run_user_pipeline,
)
from .vector_store import (
    create_collection,
    derive_point_id,
    get_collection_info,
    get_qdrant_client,
    sanitize_collection_name,
    upsert_embeddings,
    validate_collection,
)
from .s3_upload import (
    build_s3_key,
    cleanup_staged_s3_artifact,
    get_s3_client,
    promote_staged_s3_artifact,
    upload_directory_to_s3,
    upload_file_to_s3,
    upload_persistence_to_s3,
    verify_s3_objects,
)
try:
    from .loader import (
        load_directory,
        load_file,
        load_urls,
        stable_source_id,
    )
except ImportError:
    # When ingestion is imported as a top-level package and loader uses parent-level relative imports
    pass


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





