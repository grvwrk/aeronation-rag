"""
User Pipeline Orchestrator: Single user/request ingestion execution.

Coordinates synchronous, single-request ingestion:
1. One user / one request: HTTP or API initiated.
2. Temporary workspace: Intermediate artifacts live in an isolated temporary directory.
3. Strict zero silent loss: Chunks that fail embedding or upsert immediately fail the pipeline.
4. Guaranteed cleanup: Temporary workspace is deleted on success, stage failure, or exception.
5. No system retries: Synchronous execution fails fast with structured error info.
6. Dynamic collection type: Passed per request or read from config, validated by validator.py.
7. Uncompromised persistence compatibility: Output remains 100% compatible with LlamaIndex persist/<collection_type>/.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    CleanerError,
    ChunkerError,
    EmbedderError,
    LoaderError,
    PersistenceError,
    PipelineError,
    StageExecutionError,
    VectorStoreError,
)
from models import (
    IngestionStatus,
    RawDoc,
    S3UploadConfig,
    StageCounts,
    UserPipelineConfig,
    UserPipelineResult,
)
from validator import validate_collection_type, validate_local_artifact

from .chunker import chunk_documents
from .cleaner import clean_documents
from .embedder import embed_chunks
from .loader import _is_url, load_file, load_urls
from .persistence import (
    generate_ingestion_id,
    persist_collection,
    validate_persisted_artifact,
)
from .s3_upload import upload_persistence_to_s3
from .vector_store import (
    create_collection,
    get_qdrant_client,
    upsert_embeddings,
)

logger = logging.getLogger(__name__)


def run_user_pipeline(
    source: Path | str | bytes | list[str] | list[RawDoc] | None = None,
    *,
    filename: str | None = None,
    collection_type: str | None = None,
    ingestion_id: str | None = None,
    config: UserPipelineConfig | None = None,
    client: Any | None = None,
    embedding_engine: Any | None = None,
    persist_dir: Path | str | None = None,
    persist_artifact: bool = True,
    upload_to_s3: bool = False,
    s3_config: S3UploadConfig | None = None,
    s3_client: Any | None = None,
    allow_in_memory: bool = False,
    raise_on_error: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> UserPipelineResult:
    """Execute a single-user data ingestion request with temporary workspace isolation.

    Parameters:
        source: File path, raw bytes, URL, list of URLs, or list of RawDocs.
        filename: Original filename if source is provided as raw bytes.
        collection_type: Target collection name (e.g. 'rag_llm', 'aerospace_manuals').
        ingestion_id: Optional custom ingestion identifier.
        config: Optional UserPipelineConfig overrides.
        client: Optional QdrantClient instance (for testing / mocking).
        embedding_engine: Optional EmbeddingEngine instance (for testing / mocking).
        persist_dir: Base directory to store production persist files (default: 'persist').
        persist_artifact: Whether to copy validated persistence to target persist directory.
        upload_to_s3: Whether to upload validated artifact to S3 (default: False for user pipeline).
        s3_config: Optional S3UploadConfig if S3 upload is requested.
        s3_client: Optional boto3 S3 client (for testing / mocking).
        allow_in_memory: Whether to allow in-memory Qdrant instance for test suites.
        raise_on_error: If True, re-raises pipeline exceptions instead of returning failed result.
        extra_metadata: Optional dictionary of metadata merged into loaded documents.

    Returns:
        UserPipelineResult: Comprehensive telemetry, stage status, and counts.
    """
    start_time = time.perf_counter()
    cfg = config if config is not None else UserPipelineConfig.from_env(collection_type=collection_type)
    counts = StageCounts()
    current_stage = "init"
    target_col = str(collection_type or cfg.collection_type or "unknown")
    ing_id = str(ingestion_id or cfg.ingestion_id or "unknown")
    last_persistence_res = None
    last_vector_res = None
    last_s3_res = None

    # Create isolated temporary workspace for the lifecycle of this request
    with tempfile.TemporaryDirectory(prefix="user_ingestion_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        try:
            # 1. Validate collection type via validation layer
            current_stage = "init"
            raw_col = collection_type or cfg.collection_type
            target_col = validate_collection_type(raw_col)

            # 2. Ingestion ID (generated once per request)
            ing_id = ingestion_id or cfg.ingestion_id or generate_ingestion_id(target_col)

            logger.info(
                "Starting user ingestion request for '%s' (ingestion_id=%s)",
                target_col,
                ing_id,
            )

            # -----------------------------------------------------------------
            # Stage 1: Loader (load single file, URLs, bytes, or in-memory docs)
            # -----------------------------------------------------------------
            current_stage = "loader"
            logger.debug("[%s] Loading user source", current_stage)

            if isinstance(source, bytes):
                fname = filename or "uploaded_source.pdf"
                upload_file_path = temp_dir / "uploads" / fname
                upload_file_path.parent.mkdir(parents=True, exist_ok=True)
                upload_file_path.write_bytes(source)
                docs = load_file(upload_file_path, extra_metadata=extra_metadata)
                counts.docs_loaded = len(docs)
                counts.files_seen = 1

            elif isinstance(source, (str, Path)):
                src_str = str(source)
                if _is_url(src_str):
                    docs = load_file(src_str, extra_metadata=extra_metadata)
                else:
                    docs = load_file(Path(src_str), extra_metadata=extra_metadata)
                counts.docs_loaded = len(docs)
                counts.files_seen = 1

            elif isinstance(source, list) and len(source) > 0 and isinstance(source[0], str):
                docs = load_urls(source, strict=True, counts=counts, extra_metadata=extra_metadata)

            elif isinstance(source, list) and (len(source) == 0 or isinstance(source[0], RawDoc)):
                docs = list(source)
                counts.docs_loaded = len(docs)
                counts.files_seen = len(set(doc.source_id for doc in docs)) if docs else 0

            elif source is None:
                docs = []

            else:
                raise StageExecutionError(
                    f"Unsupported source type '{type(source).__name__}'",
                    stage="loader",
                )

            if not docs:
                raise StageExecutionError("Loader extracted 0 documents from source", stage="loader")

            logger.info("[%s] Loaded %d document pages", current_stage, len(docs))

            # -----------------------------------------------------------------
            # Stage 2: Cleaner (strip noise, normalize text)
            # -----------------------------------------------------------------
            current_stage = "cleaner"
            cleaned_docs = clean_documents(docs, counts=counts)
            if not cleaned_docs and docs:
                raise StageExecutionError("Cleaner discarded all documents", stage="cleaner")

            logger.info("[%s] Cleaned %d documents (discarded: %d)", current_stage, counts.docs_cleaned, counts.docs_discarded)

            # -----------------------------------------------------------------
            # Stage 3: Chunker (split into overlapping token-sized chunks)
            # -----------------------------------------------------------------
            current_stage = "chunker"
            chunks = chunk_documents(cleaned_docs, config=cfg.chunking_config, counts=counts)
            if not chunks and cleaned_docs:
                raise StageExecutionError("Chunker produced 0 chunks from valid cleaned documents", stage="chunker")

            logger.info("[%s] Produced %d chunks", current_stage, len(chunks))

            # -----------------------------------------------------------------
            # Stage 4: Embedder (generate dense and sparse embeddings)
            # -----------------------------------------------------------------
            current_stage = "embedder"
            emb_res = embed_chunks(
                chunks,
                config=cfg.embedding_config,
                engine=embedding_engine,
                counts=counts,
                strict=True,
            )

            # Zero silent loss: if any chunk failed embedding, fail the request
            if emb_res.failed_chunks or emb_res.failed_count > 0:
                raise StageExecutionError(
                    f"Embedding failed: {emb_res.failed_count} of {emb_res.total_chunks} chunks failed embedding",
                    stage="embedder",
                )

            logger.info("[%s] Generated %d embeddings", current_stage, emb_res.successful_chunks)

            # -----------------------------------------------------------------
            # Stage 5: Vector Store (Qdrant collection creation and vector upsert)
            # -----------------------------------------------------------------
            current_stage = "vector_store"
            q_client = get_qdrant_client(cfg.vector_config, client=client, allow_in_memory=allow_in_memory)
            create_collection(q_client, cfg.vector_config)
            v_res = upsert_embeddings(
                q_client,
                emb_res.embedded_chunks,
                config=cfg.vector_config,
                counts=counts,
                strict=True,
            )
            last_vector_res = v_res

            if v_res.vectors_failed > 0:
                raise StageExecutionError(
                    f"Vector store upsert failed: {v_res.vectors_failed} vectors failed",
                    stage="vector_store",
                )

            logger.info("[%s] Upserted %d vectors", current_stage, v_res.vectors_inserted)

            # -----------------------------------------------------------------
            # Stage 6: Persistence (write standard LlamaIndex files in temp dir)
            # -----------------------------------------------------------------
            current_stage = "persistence"
            temp_persist_dir = temp_dir / "persist"
            p_res = persist_collection(
                collection_type=target_col,
                ingestion_id=ing_id,
                vector_config=cfg.vector_config,
                embedding_config=cfg.embedding_config,
                chunking_config=cfg.chunking_config,
                counts=counts,
                persist_dir=temp_persist_dir,
            )
            last_persistence_res = p_res

            # -----------------------------------------------------------------
            # Stage 7: Validation (validate manifest and artifact completeness)
            # -----------------------------------------------------------------
            current_stage = "validator"
            validate_persisted_artifact(
                p_res.persist_dir,
                expected_ingestion_id=ing_id,
                expected_collection_type=target_col,
            )
            logger.info("[%s] Validated persistence artifact for '%s'", current_stage, target_col)

            # -----------------------------------------------------------------
            # Stage 8: Target Promotion (copy validated files to final persist dir)
            # -----------------------------------------------------------------
            target_persist_base = Path(persist_dir or cfg.persist_dir)
            target_col_dir = target_persist_base / target_col

            if persist_artifact:
                target_col_dir.mkdir(parents=True, exist_ok=True)
                for file_path in Path(p_res.persist_dir).glob("*"):
                    if file_path.is_file():
                        shutil.copy2(file_path, target_col_dir / file_path.name)

                # Re-validate the promoted production artifact
                validate_persisted_artifact(
                    target_col_dir,
                    expected_ingestion_id=ing_id,
                    expected_collection_type=target_col,
                )
                logger.info("Persisted validated artifact to %s", target_col_dir)

            # -----------------------------------------------------------------
            # Stage 9: Optional S3 Upload (only if explicitly enabled)
            # -----------------------------------------------------------------
            if upload_to_s3 or cfg.upload_to_s3:
                current_stage = "s3_upload"
                s3_cfg = s3_config or cfg.s3_config or S3UploadConfig.from_env(collection_type=target_col)
                s3_src = target_col_dir if persist_artifact else Path(p_res.persist_dir)
                last_s3_res = upload_persistence_to_s3(
                    s3_src,
                    config=s3_cfg,
                    s3_client=s3_client,
                    collection_type=target_col,
                    ingestion_id=ing_id,
                )
                logger.info("Uploaded artifact to S3 bucket '%s'", s3_cfg.bucket_name)

            duration = time.perf_counter() - start_time
            logger.info(
                "User ingestion request completed successfully in %.2fs: %d docs, %d chunks, %d vectors",
                duration,
                counts.docs_loaded,
                counts.chunks_created,
                counts.vectors_inserted,
            )

            return UserPipelineResult(
                success=True,
                status=IngestionStatus.COMPLETED,
                ingestion_id=ing_id,
                collection_type=target_col,
                stage="completed",
                documents=counts.docs_loaded,
                chunks=counts.chunks_created,
                embedded=counts.embeddings_generated,
                failed_embeddings=counts.chunks_failed,
                persisted=persist_artifact,
                validated=True,
                counts=counts,
                persistence_result=p_res,
                vector_result=v_res,
                s3_result=last_s3_res,
                error=None,
                duration_seconds=duration,
            )

        except Exception as exc:
            duration = time.perf_counter() - start_time
            err_msg = str(exc)
            logger.error(
                "User ingestion request failed at stage '%s' after %.2fs: %s",
                current_stage,
                duration,
                err_msg,
            )

            if raise_on_error:
                raise

            return UserPipelineResult(
                success=False,
                status=IngestionStatus.FAILED,
                ingestion_id=ing_id,
                collection_type=target_col,
                stage=current_stage,
                documents=counts.docs_loaded,
                chunks=counts.chunks_created,
                embedded=counts.embeddings_generated,
                failed_embeddings=counts.chunks_failed,
                persisted=False,
                validated=False,
                counts=counts,
                persistence_result=last_persistence_res,
                vector_result=last_vector_res,
                s3_result=last_s3_res,
                error=err_msg,
                duration_seconds=duration,
            )
