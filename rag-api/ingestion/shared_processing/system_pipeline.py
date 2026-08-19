"""System Pipeline: Orchestration with bounded re-ingestion and retry recovery.

Implements the end-to-end orchestration layer for the Aeronation-RAG data ingestion pipeline:
1. Bounded retries: Runs at most MAX_REINGESTION_ATTEMPTS (default: 3). Exactly attempt 1 (initial), attempt 2 (first retry), attempt 3 (second retry).
2. Idempotent restart: On stage failure (e.g. partway embedding error), cleans up staging artifacts and restarts safely without duplicating production data.
3. Durable state tracking: Persists IngestionState (PENDING, PROCESSING, FAILED, RETRY_EXHAUSTED, COMPLETED) to disk.
4. Strict promotion safety: Only completed and validated ingestions are marked COMPLETED. FAILED or RETRY_EXHAUSTED are never promoted.
5. No infinite retry loop: Hard stop when attempts reach max_attempts.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    PipelineError,
    PipelineExecutionError,
    RetryExhaustedError,
    StageExecutionError,
)
from models import (
    IngestionState,
    IngestionStatus,
    PipelineConfig,
    PipelineResult,
    RawDoc,
    StageCounts,
)
from .chunker import chunk_documents
from .cleaner import clean_documents
from .embedder import embed_chunks
from .loader import load_directory, load_urls
from .persistence import (
    generate_ingestion_id,
    persist_collection,
    save_ingestion_state,
    validate_persisted_artifact,
)
from .vector_store import (
    create_collection,
    get_qdrant_client,
    upsert_embeddings,
)

logger = logging.getLogger(__name__)


def cleanup_staging_artifacts(base_persist_dir: str | Path, collection_type: str, ingestion_id: str) -> None:
    """Clean up temporary staging directories associated with an ingestion attempt."""
    staging_dir = Path(base_persist_dir) / ".tmp" / f"{collection_type}_{ingestion_id}"
    if staging_dir.exists():
        try:
            shutil.rmtree(staging_dir, ignore_errors=True)
            logger.debug("Cleaned up staging directory: %s", staging_dir)
        except Exception as exc:
            logger.warning("Failed to clean up staging directory %s: %s", staging_dir, exc)


def run_system_pipeline(
    config: PipelineConfig | None = None,
    *,
    client: Any | None = None,
    raw_docs: list[RawDoc] | None = None,
    embedding_engine: Any | None = None,
    raise_on_exhaustion: bool = False,
    allow_in_memory: bool = False,
) -> PipelineResult:
    """Execute the full data ingestion pipeline with bounded re-ingestion and retry handling.

    - Attempt 1: Initial ingestion attempt.
    - Attempt 2: First retry on failure.
    - Attempt 3: Second retry on failure.
    - When attempt reaches max_attempts and fails: transitions to RETRY_EXHAUSTED and stops.
    """
    cfg = config if config is not None else PipelineConfig.from_env()
    ing_id = cfg.ingestion_id or generate_ingestion_id(cfg.collection_type)
    max_attempts = max(1, cfg.max_attempts)

    # Initialize state tracker
    state = IngestionState(
        ingestion_id=ing_id,
        version=ing_id,
        collection_type=cfg.collection_type,
        status=IngestionStatus.PENDING,
        current_stage="init",
        attempt=1,
        max_attempts=max_attempts,
    )

    last_error: str | None = None
    last_counts = StageCounts()
    last_persistence_res = None
    last_vector_res = None

    for attempt in range(1, max_attempts + 1):
        state.attempt = attempt
        state.mark_stage("init", IngestionStatus.PROCESSING)
        save_ingestion_state(state, cfg.persist_dir)

        counts = StageCounts()
        current_stage = "init"

        logger.info(
            "Starting ingestion pipeline for '%s' (ingestion_id=%s, attempt %d of %d)",
            cfg.collection_type,
            ing_id,
            attempt,
            max_attempts,
        )

        try:
            # Stage 1: Loader
            current_stage = "loader"
            state.mark_stage(current_stage, IngestionStatus.PROCESSING)

            if raw_docs is not None:
                docs = list(raw_docs)
                counts.docs_loaded = len(docs)
                counts.files_seen = len(set(doc.source_id for doc in docs))
            elif cfg.urls:
                if load_urls is None:
                    raise StageExecutionError("load_urls is unavailable in this environment", stage="loader")
                docs = load_urls(cfg.urls, counts=counts)
            elif cfg.data_dir:
                if load_directory is None:
                    raise StageExecutionError("load_directory is unavailable in this environment", stage="loader")
                docs = load_directory(cfg.data_dir, counts=counts)
            else:
                docs = []

            logger.info("[%s] Loaded %d documents from %d files", current_stage, counts.docs_loaded, counts.files_seen)

            # Stage 2: Cleaner
            current_stage = "cleaner"
            state.mark_stage(current_stage, IngestionStatus.PROCESSING)
            cleaned_docs = clean_documents(docs, counts=counts)
            logger.info("[%s] Cleaned %d documents (discarded: %d)", current_stage, counts.docs_cleaned, counts.docs_discarded)

            # Stage 3: Chunker
            current_stage = "chunker"
            state.mark_stage(current_stage, IngestionStatus.PROCESSING)
            chunks = chunk_documents(cleaned_docs, config=cfg.chunking_config, counts=counts)
            logger.info("[%s] Created %d chunks", current_stage, counts.chunks_created)

            if not chunks and docs:
                raise StageExecutionError("Chunker produced 0 chunks from valid documents", stage="chunker")

            # Stage 4: Embedder (strict mode to catch any batch/chunk failures)
            current_stage = "embedder"
            state.mark_stage(current_stage, IngestionStatus.PROCESSING)
            emb_result = embed_chunks(chunks, config=cfg.embedding_config, engine=embedding_engine, counts=counts, strict=True)
            logger.info("[%s] Generated %d embeddings", current_stage, emb_result.successful_chunks)

            if emb_result.failed_chunks:
                raise StageExecutionError(
                    f"{len(emb_result.failed_chunks)} chunks failed dense/sparse embedding",
                    stage="embedder",
                )

            # Stage 5: Vector Store
            current_stage = "vector_store"
            state.mark_stage(current_stage, IngestionStatus.PROCESSING)
            q_client = get_qdrant_client(cfg.vector_config, client=client, allow_in_memory=allow_in_memory)
            create_collection(q_client, cfg.vector_config)
            v_res = upsert_embeddings(
                q_client,
                emb_result.embedded_chunks,
                config=cfg.vector_config,
                counts=counts,
                strict=True,
            )
            last_vector_res = v_res
            logger.info("[%s] Upserted %d vectors (failed: %d)", current_stage, v_res.vectors_inserted, v_res.vectors_failed)

            if v_res.vectors_failed > 0:
                raise StageExecutionError(
                    f"{v_res.vectors_failed} vectors failed during vector store upsert",
                    stage="vector_store",
                )

            # Stage 6: Persistence
            current_stage = "persistence"
            state.mark_stage(current_stage, IngestionStatus.PROCESSING)
            p_res = persist_collection(
                collection_type=cfg.collection_type,
                ingestion_id=ing_id,
                vector_config=cfg.vector_config,
                embedding_config=cfg.embedding_config,
                chunking_config=cfg.chunking_config,
                counts=counts,
                state=state,
                persist_dir=cfg.persist_dir,
            )
            last_persistence_res = p_res

            # Stage 7: Validation
            current_stage = "validator"
            state.mark_stage(current_stage, IngestionStatus.VALIDATING)
            validate_persisted_artifact(p_res.persist_dir, expected_ingestion_id=ing_id, expected_collection_type=cfg.collection_type)

            # Stage 8: Completed
            state.mark_completed()
            manifest_path = Path(p_res.persist_dir) / "manifest.json"
            if manifest_path.is_file():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    m_data["status"] = IngestionStatus.COMPLETED
                    m_data["updated_at"] = state.updated_at
                    with open(manifest_path, "w", encoding="utf-8") as f:
                        json.dump(m_data, f, indent=2)
                except Exception as exc:
                    logger.warning("Failed updating manifest status to COMPLETED: %s", exc)

            save_ingestion_state(state, cfg.persist_dir)

            logger.info(
                "Ingestion pipeline completed successfully for '%s' (ingestion_id=%s) on attempt %d",
                cfg.collection_type,
                ing_id,
                attempt,
            )

            return PipelineResult(
                ingestion_id=ing_id,
                collection_type=cfg.collection_type,
                status=IngestionStatus.COMPLETED,
                attempt=attempt,
                max_attempts=max_attempts,
                counts=counts,
                persistence_result=p_res,
                vector_result=v_res,
                error=None,
                success=True,
            )

        except Exception as exc:
            last_error = str(exc)
            last_counts = counts
            cleanup_staging_artifacts(cfg.persist_dir, cfg.collection_type, ing_id)

            logger.error(
                "Ingestion pipeline failed at stage '%s' on attempt %d of %d: %s",
                current_stage,
                attempt,
                max_attempts,
                exc,
            )

            if attempt >= max_attempts:
                # Retry budget exhausted
                state.mark_retry_exhausted(stage=current_stage, error=last_error)
                save_ingestion_state(state, cfg.persist_dir)

                logger.error(
                    "Ingestion pipeline for '%s' exhausted all %d attempts. Marking as %s.",
                    cfg.collection_type,
                    max_attempts,
                    IngestionStatus.RETRY_EXHAUSTED,
                )

                if raise_on_exhaustion:
                    raise RetryExhaustedError(
                        f"Pipeline failed at stage '{current_stage}' after {max_attempts} attempts: {last_error}",
                        stage=current_stage,
                        ingestion_id=ing_id,
                        attempt=attempt,
                    ) from exc

                return PipelineResult(
                    ingestion_id=ing_id,
                    collection_type=cfg.collection_type,
                    status=IngestionStatus.RETRY_EXHAUSTED,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    counts=last_counts,
                    persistence_result=last_persistence_res,
                    vector_result=last_vector_res,
                    error=last_error,
                    success=False,
                )

            # Not yet exhausted: mark FAILED and retry
            state.mark_failed(stage=current_stage, error=last_error)
            save_ingestion_state(state, cfg.persist_dir)
            logger.info("Preparing for retry attempt %d...", attempt + 1)

    # Fallback return in case loop completes
    return PipelineResult(
        ingestion_id=ing_id,
        collection_type=cfg.collection_type,
        status=IngestionStatus.RETRY_EXHAUSTED,
        attempt=max_attempts,
        max_attempts=max_attempts,
        counts=last_counts,
        persistence_result=last_persistence_res,
        vector_result=last_vector_res,
        error=last_error or "Max attempts reached",
        success=False,
    )


def main() -> None:
    """CLI entry point for running the system ingestion pipeline."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run the Aeronation RAG System Ingestion Pipeline")
    parser.add_argument("--data-dir", default=None, help="Directory with source documents (e.g. PDFs)")
    parser.add_argument("--urls", nargs="+", default=None, help="List of URLs to ingest")
    parser.add_argument("--collection", dest="collection_type", default=None, help="Collection type / name (e.g. rag_llm)")
    parser.add_argument("--ingestion-id", default=None, help="Custom ingestion identifier")
    parser.add_argument("--persist-dir", default=None, help="Persistence base directory")
    parser.add_argument("--max-attempts", type=int, default=None, help="Maximum retry attempts")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap")
    args = parser.parse_args()

    config = PipelineConfig.from_env(collection_type=args.collection_type)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.urls:
        config.urls = args.urls
    if args.ingestion_id:
        config.ingestion_id = args.ingestion_id
    if args.persist_dir:
        config.persist_dir = args.persist_dir
    if args.max_attempts is not None:
        config.max_attempts = args.max_attempts
    if args.chunk_size is not None:
        config.chunking_config.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        config.chunking_config.chunk_overlap = args.chunk_overlap

    logger.info("Executing system pipeline with collection_type=%s, data_dir=%s", config.collection_type, config.data_dir)
    result = run_system_pipeline(config)
    if result.success:
        logger.info("System pipeline completed successfully! Ingestion ID: %s", result.ingestion_id)
        sys.exit(0)
    else:
        logger.error("System pipeline failed: %s", result.error)
        sys.exit(1)


if __name__ == "__main__":
    main()

