"""Unit tests for system pipeline orchestration, bounded retry, and stage failure recovery.
"""

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import RetryExhaustedError, StageExecutionError
from ingestion.shared_processing.embedder import EmbeddingEngine
from ingestion.shared_processing.persistence import (
    load_ingestion_state,
    load_manifest,
    persist_collection,
    validate_persisted_artifact,
)
from ingestion.shared_processing.system_pipeline import run_system_pipeline
from models import (
    Chunk,
    ChunkingConfig,
    EmbeddingConfig,
    IngestionState,
    IngestionStatus,
    PersistenceConfig,
    PipelineConfig,
    RawDoc,
    VectorStoreConfig,
)
from qdrant_client import QdrantClient


class MockDenseModel:
    """Deterministic mock dense embedding model."""

    def __init__(self, dim: int = 384, fail_on_text: str | None = None) -> None:
        self.dim = dim
        self.fail_on_text = fail_on_text
        self.call_count = 0

    def embed(self, texts: list[str]):
        self.call_count += 1
        for t in texts:
            if self.fail_on_text and self.fail_on_text in t:
                raise RuntimeError(f"Simulated dense embedding failure on: {t}")
            yield [0.1] * self.dim


class MockSparseModel:
    """Deterministic mock sparse embedding model."""

    def __init__(self, fail_on_text: str | None = None) -> None:
        self.fail_on_text = fail_on_text
        self.call_count = 0

    def embed(self, texts: list[str]):
        self.call_count += 1
        for t in texts:
            if self.fail_on_text and self.fail_on_text in t:
                raise RuntimeError(f"Simulated sparse embedding failure on: {t}")
            mock_emb = MagicMock()
            mock_emb.indices = [1, 2, 3]
            mock_emb.values = [0.5, 0.6, 0.7]
            yield mock_emb


class SystemPipelineUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.qdrant_client = QdrantClient(location=":memory:")
        self.sample_docs = [
            RawDoc(
                text="The lift force on an airfoil is proportional to the dynamic pressure and wing area.",
                source_id="aero_doc_1.pdf",
                metadata={"page_num": 1},
            ),
            RawDoc(
                text="Induced drag occurs due to the creation of wingtip vortices in finite span wings.",
                source_id="aero_doc_2.pdf",
                metadata={"page_num": 2},
            ),
        ]
        self.config = PipelineConfig(
            collection_type="rag_test",
            ingestion_id="test_ingestion_001",
            max_attempts=3,
            persist_dir=self.temp_dir,
            chunking_config=ChunkingConfig(chunk_size=128, chunk_overlap=16),
            embedding_config=EmbeddingConfig(
                dense_model="mock-dense",
                sparse_model="mock-sparse",
                expected_dimension=384,
            ),
            vector_config=VectorStoreConfig(
                collection_name="rag_test",
                dimension=384,
                distance="Cosine",
                enable_hybrid=True,
            ),
        )
        self.mock_engine = EmbeddingEngine(self.config.embedding_config)
        self.mock_engine._dense_model = MockDenseModel(dim=384)
        self.mock_engine._sparse_model = MockSparseModel()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Test 1 — successful ingestion on attempt 1
    def test_successful_ingestion_attempt_1(self):
        result = run_system_pipeline(
            self.config,
            client=self.qdrant_client,
            raw_docs=self.sample_docs,
            embedding_engine=self.mock_engine,
            allow_in_memory=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, IngestionStatus.COMPLETED)
        self.assertEqual(result.attempt, 1)
        self.assertGreater(result.counts.chunks_created, 0)
        self.assertGreater(result.counts.vectors_inserted, 0)

        # Verify persisted state
        state = load_ingestion_state(self.temp_dir, "rag_test")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, IngestionStatus.COMPLETED)
        self.assertEqual(state.attempt, 1)

        # Verify manifest
        manifest = load_manifest(Path(self.temp_dir) / "rag_test")
        self.assertEqual(manifest["status"], IngestionStatus.COMPLETED)
        self.assertEqual(manifest["attempt"], 1)

    # 2. Test 2 — embedding fails on attempt 1, succeeds on attempt 2
    def test_embedding_fails_once_and_succeeds_on_retry(self):
        attempt_counter = [0]

        from ingestion.shared_processing import embedder

        orig_embed_chunks = embedder.embed_chunks

        def mock_embed_chunks(chunks, *args, **kwargs):
            attempt_counter[0] += 1
            if attempt_counter[0] == 1:
                raise RuntimeError("Simulated transient GPU/network failure during embedding batch")
            return orig_embed_chunks(chunks, *args, **kwargs)

        with patch("ingestion.shared_processing.system_pipeline.embed_chunks", side_effect=mock_embed_chunks):
            result = run_system_pipeline(
                self.config,
                client=self.qdrant_client,
                raw_docs=self.sample_docs,
                embedding_engine=self.mock_engine,
                allow_in_memory=True,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, IngestionStatus.COMPLETED)
        self.assertEqual(result.attempt, 2)
        self.assertEqual(attempt_counter[0], 2)

        # Verify state indicates completed at attempt 2
        state = load_ingestion_state(self.temp_dir, "rag_test")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, IngestionStatus.COMPLETED)
        self.assertEqual(state.attempt, 2)

    # 3. Test 3 & 4 — embedding fails on every attempt, max attempts respected, RETRY_EXHAUSTED
    def test_embedding_fails_all_attempts_exhausted(self):
        attempt_counter = [0]

        def mock_embed_chunks(chunks, *args, **kwargs):
            attempt_counter[0] += 1
            raise RuntimeError("Permanent dense embedding server down")

        with patch("ingestion.shared_processing.system_pipeline.embed_chunks", side_effect=mock_embed_chunks):
            result = run_system_pipeline(
                self.config,
                client=self.qdrant_client,
                raw_docs=self.sample_docs,
                embedding_engine=self.mock_engine,
                allow_in_memory=True,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, IngestionStatus.RETRY_EXHAUSTED)
        self.assertEqual(result.attempt, 3)
        self.assertEqual(attempt_counter[0], 3)
        self.assertIn("Permanent dense embedding", result.error)

        # Verify state
        state = load_ingestion_state(self.temp_dir, "rag_test")
        self.assertIsNotNone(state)
        self.assertEqual(state.status, IngestionStatus.RETRY_EXHAUSTED)
        self.assertEqual(state.failed_stage, "embedder")
        self.assertEqual(state.attempt, 3)

    # 4. Test raise_on_exhaustion
    def test_raise_on_exhaustion(self):
        with patch("ingestion.shared_processing.system_pipeline.embed_chunks", side_effect=RuntimeError("Permanent failure")):
            with self.assertRaises(RetryExhaustedError) as ctx:
                run_system_pipeline(
                    self.config,
                    client=self.qdrant_client,
                    raw_docs=self.sample_docs,
                    embedding_engine=self.mock_engine,
                    raise_on_exhaustion=True,
                    allow_in_memory=True,
                )
            self.assertEqual(ctx.exception.stage, "embedder")
            self.assertEqual(ctx.exception.attempt, 3)

    # 5. Test 5 — partial failure during embedding cleans up staging artifacts
    def test_partial_embedding_failure_cleans_up_staging(self):
        staging_path = Path(self.temp_dir) / ".tmp" / f"rag_test_{self.config.ingestion_id}"

        def mock_failing_embed(chunks, *args, **kwargs):
            staging_path.mkdir(parents=True, exist_ok=True)
            (staging_path / "partial_test.txt").write_text("temporary data", encoding="utf-8")
            raise RuntimeError("Failure partway through embedding")

        with patch("ingestion.shared_processing.system_pipeline.embed_chunks", side_effect=mock_failing_embed):
            result = run_system_pipeline(
                self.config,
                client=self.qdrant_client,
                raw_docs=self.sample_docs,
                embedding_engine=self.mock_engine,
                allow_in_memory=True,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, IngestionStatus.RETRY_EXHAUSTED)
        # Staging directory must be cleaned up
        self.assertFalse(staging_path.exists())

    # 6. Test 6 — validation failure triggers bounded retry
    def test_validation_failure_triggers_retry(self):
        call_count = [0]

        from ingestion.shared_processing import persistence

        orig_validate = persistence.validate_persisted_artifact

        def mock_validate(persist_dir, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated manifest JSON corruption or validation failure on attempt 1")
            return orig_validate(persist_dir, *args, **kwargs)

        with patch("ingestion.shared_processing.system_pipeline.validate_persisted_artifact", side_effect=mock_validate):
            result = run_system_pipeline(
                self.config,
                client=self.qdrant_client,
                raw_docs=self.sample_docs,
                embedding_engine=self.mock_engine,
                allow_in_memory=True,
            )

        self.assertTrue(result.success)
        self.assertEqual(result.status, IngestionStatus.COMPLETED)
        self.assertEqual(result.attempt, 2)

    # 7. Test 7 — retry does not duplicate production vectors or create corrupted files
    def test_retry_leaves_clean_final_artifact(self):
        call_count = [0]

        def mock_sometimes_fail(chunks, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Temporary glitch on attempt 1")
            from ingestion.shared_processing.embedder import embed_chunks
            return embed_chunks(chunks, *args, **kwargs)

        with patch("ingestion.shared_processing.system_pipeline.embed_chunks", side_effect=mock_sometimes_fail):
            result = run_system_pipeline(
                self.config,
                client=self.qdrant_client,
                raw_docs=self.sample_docs,
                embedding_engine=self.mock_engine,
                allow_in_memory=True,
            )

        self.assertTrue(result.success)
        target_dir = Path(self.temp_dir) / "rag_test"
        self.assertTrue(validate_persisted_artifact(target_dir))

    # 8. Test 8 — unrelated production data remains untouched when another ingestion fails
    def test_failed_retry_does_not_modify_existing_production_data(self):
        # 1. Create a prior valid collection artifact in production path
        prior_config = PipelineConfig(
            collection_type="production_col",
            ingestion_id="v_prod_001",
            persist_dir=self.temp_dir,
            vector_config=VectorStoreConfig(collection_name="production_col"),
        )
        persist_collection(
            collection_type="production_col",
            ingestion_id="v_prod_001",
            vector_config=prior_config.vector_config,
            persist_dir=self.temp_dir,
        )

        prod_dir = Path(self.temp_dir) / "production_col"
        self.assertTrue(validate_persisted_artifact(prod_dir, expected_ingestion_id="v_prod_001"))

        # 2. Run a new failing ingestion for a different collection
        failing_config = PipelineConfig(
            collection_type="new_failing_col",
            ingestion_id="v_fail_002",
            max_attempts=2,
            persist_dir=self.temp_dir,
        )

        with patch("ingestion.shared_processing.system_pipeline.clean_documents", side_effect=RuntimeError("Fatal clean error")):
            result = run_system_pipeline(
                failing_config,
                client=self.qdrant_client,
                raw_docs=self.sample_docs,
                embedding_engine=self.mock_engine,
                allow_in_memory=True,
            )

        self.assertFalse(result.success)
        self.assertEqual(result.status, IngestionStatus.RETRY_EXHAUSTED)

        # 3. Verify prior production collection is 100% intact and unchanged
        self.assertTrue(validate_persisted_artifact(prod_dir, expected_ingestion_id="v_prod_001"))
        manifest = load_manifest(prod_dir)
        self.assertEqual(manifest["ingestion_id"], "v_prod_001")


if __name__ == "__main__":
    unittest.main()
