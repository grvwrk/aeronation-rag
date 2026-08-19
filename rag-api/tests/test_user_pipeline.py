"""
Unit tests for the User Pipeline Orchestrator (Single-User / Request Ingestion).
"""

import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    CleanerError,
    ChunkerError,
    EmbedderError,
    IncompleteArtifactError,
    LoaderError,
    ManifestError,
    PersistenceError,
    PipelineError,
    StageExecutionError,
    VectorStoreError,
)
from ingestion.shared_processing.user_pipeline import run_user_pipeline
from models import (
    ChunkingConfig,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingResult,
    FailedChunk,
    IngestionStatus,
    RawDoc,
    SparseEmbeddingData,
    StageCounts,
    UserPipelineConfig,
    UserPipelineResult,
    VectorStoreConfig,
    VectorStoreResult,
)
from validator import validate_collection_type, validate_local_artifact


class MockEmbeddingEngine:
    """Deterministic mock embedding engine for unit tests."""

    def __init__(self, dimension: int = 384, fail_on_text: str | None = None) -> None:
        self.dimension = dimension
        self.fail_on_text = fail_on_text
        self.embed_calls: list[str] = []

    def embed_dense_batch(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.extend(texts)
        results = []
        for text in texts:
            if self.fail_on_text and self.fail_on_text in text:
                raise EmbedderError(source="embedder", reason=f"Simulated dense embedding failure on: {text}")
            results.append([0.05] * self.dimension)
        return results

    def embed_sparse_batch(self, texts: list[str]) -> list[SparseEmbeddingData | None]:
        return [SparseEmbeddingData(indices=[1, 5, 10], values=[0.2, 0.4, 0.6]) for _ in texts]



class MockQdrantClient:
    """Deterministic in-memory mock Qdrant client."""

    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.points: dict[str, list[Any]] = {}
        self.fail_upsert = False

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def get_collection(self, collection_name: str) -> Any:
        if collection_name not in self.collections:
            raise RuntimeError(f"Collection {collection_name} not found")
        mock_info = MagicMock()
        mock_info.config.params.vectors = {
            "text-dense": MagicMock(size=384, distance=MagicMock(value="Cosine"))
        }
        return mock_info

    def create_collection(self, collection_name: str, **kwargs: Any) -> None:
        self.collections.add(collection_name)
        if collection_name not in self.points:
            self.points[collection_name] = []

    def upsert(self, collection_name: str, points: list[Any], **kwargs: Any) -> None:
        if self.fail_upsert:
            raise RuntimeError("Simulated Qdrant cluster connection error")
        self.collections.add(collection_name)
        if collection_name not in self.points:
            self.points[collection_name] = []
        self.points[collection_name].extend(points)


class UserPipelineUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.persist_dir = Path(self.temp_dir) / "persist"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_type = "aeronautics_manuals"
        self.mock_embedder = MockEmbeddingEngine(dimension=384)
        self.mock_qdrant = MockQdrantClient()

        self.sample_docs = [
            RawDoc(
                text="The Boeing 787 Dreamliner is a wide-body airliner manufactured by Boeing.",
                source_id="b787_overview.pdf",
                metadata={"file_name": "b787_overview", "page_num": 1},
            ),
            RawDoc(
                text="Composite materials make up 50 percent of the primary structure of the 787.",
                source_id="b787_overview.pdf",
                metadata={"file_name": "b787_overview", "page_num": 2},
            ),
        ]

        self.config = UserPipelineConfig(
            collection_type=self.collection_type,
            persist_dir=str(self.persist_dir),
            persist_artifact=True,
            chunking_config=ChunkingConfig(chunk_size=128, chunk_overlap=16),
            embedding_config=EmbeddingConfig(expected_dimension=384),
            vector_config=VectorStoreConfig(collection_name=self.collection_type, dimension=384),
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Successful single-user ingestion
    def test_successful_single_user_ingestion(self) -> None:
        result = run_user_pipeline(
            source=self.sample_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.status, IngestionStatus.COMPLETED)
        self.assertEqual(result.stage, "completed")
        self.assertEqual(result.collection_type, self.collection_type)
        self.assertEqual(result.documents, 2)
        self.assertGreater(result.chunks, 0)
        self.assertEqual(result.embedded, result.chunks)
        self.assertEqual(result.failed_embeddings, 0)
        self.assertTrue(result.persisted)
        self.assertTrue(result.validated)
        self.assertIsNone(result.error)

        # Check persisted artifact in target directory
        target_dir = self.persist_dir / self.collection_type
        self.assertTrue(target_dir.is_dir())
        self.assertTrue((target_dir / "manifest.json").is_file())
        self.assertTrue((target_dir / "docstore.json").is_file())

    # 2. Temporary workspace is created during execution
    def test_temporary_workspace_created(self) -> None:
        created_temp_dirs = []
        original_tempdir = tempfile.TemporaryDirectory

        def tracking_tempdir(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory:
            instance = original_tempdir(*args, **kwargs)
            created_temp_dirs.append(Path(instance.name))
            return instance

        with patch("tempfile.TemporaryDirectory", side_effect=tracking_tempdir):
            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertTrue(res.success)
            self.assertEqual(len(created_temp_dirs), 1)

    # 3. Temporary workspace is cleaned after success
    def test_temporary_workspace_cleaned_after_success(self) -> None:
        temp_path_captured: list[Path] = []
        original_tempdir = tempfile.TemporaryDirectory

        def tracking_tempdir(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory:
            inst = original_tempdir(*args, **kwargs)
            temp_path_captured.append(Path(inst.name))
            return inst

        with patch("tempfile.TemporaryDirectory", side_effect=tracking_tempdir):
            run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )

        self.assertEqual(len(temp_path_captured), 1)
        self.assertFalse(temp_path_captured[0].exists())

    # 4. Temporary workspace is cleaned after failure
    def test_temporary_workspace_cleaned_after_failure(self) -> None:
        temp_path_captured: list[Path] = []
        original_tempdir = tempfile.TemporaryDirectory

        def tracking_tempdir(*args: Any, **kwargs: Any) -> tempfile.TemporaryDirectory:
            inst = original_tempdir(*args, **kwargs)
            temp_path_captured.append(Path(inst.name))
            return inst

        failing_embedder = MockEmbeddingEngine(fail_on_text="Boeing")

        with patch("tempfile.TemporaryDirectory", side_effect=tracking_tempdir):
            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=failing_embedder,
                allow_in_memory=True,
            )
            self.assertFalse(res.success)

        self.assertEqual(len(temp_path_captured), 1)
        self.assertFalse(temp_path_captured[0].exists())

    # 5. Loader failure returns failure with stage="loader"
    def test_loader_failure_returns_failure(self) -> None:
        res = run_user_pipeline(
            source=Path(self.temp_dir) / "nonexistent_file.pdf",
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        self.assertEqual(res.status, IngestionStatus.FAILED)
        self.assertEqual(res.stage, "loader")
        self.assertIn("does not exist", res.error.lower())

    # 6. Cleaner failure returns failure with stage="cleaner"
    def test_cleaner_failure_returns_failure(self) -> None:
        empty_docs = [RawDoc(text="   \n\t  ", source_id="empty.txt")]
        res = run_user_pipeline(
            source=empty_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        self.assertEqual(res.status, IngestionStatus.FAILED)
        self.assertEqual(res.stage, "cleaner")

    # 7. Chunker failure returns failure with stage="chunker"
    def test_chunker_failure_returns_failure(self) -> None:
        with patch("ingestion.shared_processing.user_pipeline.chunk_documents", return_value=[]):
            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertFalse(res.success)
            self.assertEqual(res.status, IngestionStatus.FAILED)
            self.assertEqual(res.stage, "chunker")

    # 8. Embedding failure returns failure with stage="embedder"
    def test_embedding_failure_returns_failure(self) -> None:
        failing_embedder = MockEmbeddingEngine(fail_on_text="Boeing")
        res = run_user_pipeline(
            source=self.sample_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=failing_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        self.assertEqual(res.status, IngestionStatus.FAILED)
        self.assertEqual(res.stage, "embedder")
        self.assertIn("embedding", res.error.lower())

    # 9. Partial embedding failure is NOT treated as success (Zero Silent Loss)
    def test_partial_embedding_failure_is_failed(self) -> None:
        with patch("ingestion.shared_processing.user_pipeline.embed_chunks") as mock_emb:
            # Simulate 10 total chunks, 9 succeeded, 1 failed
            mock_emb.return_value = EmbeddingResult(
                embedded_chunks=[],
                failed_chunks=[FailedChunk(chunk=MagicMock(), reason="Failed 1 chunk out of 10")],
                total_chunks=10,
                successful_chunks=9,
                failed_count=1,
            )

            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertFalse(res.success)
            self.assertEqual(res.status, IngestionStatus.FAILED)
            self.assertEqual(res.stage, "embedder")
            self.assertIn("1 of 10 chunks failed", res.error)

    # 10. Vector-store failure returns failure with stage="vector_store"
    def test_vector_store_failure_returns_failure(self) -> None:
        self.mock_qdrant.fail_upsert = True
        res = run_user_pipeline(
            source=self.sample_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        self.assertEqual(res.status, IngestionStatus.FAILED)
        self.assertEqual(res.stage, "vector_store")

    # 11. Persistence failure returns failure with stage="persistence"
    def test_persistence_failure_returns_failure(self) -> None:
        with patch("ingestion.shared_processing.user_pipeline.persist_collection") as mock_p:
            mock_p.side_effect = RuntimeError("Disk quota exceeded during persistence")
            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertFalse(res.success)
            self.assertEqual(res.status, IngestionStatus.FAILED)
            self.assertEqual(res.stage, "persistence")

    # 12. Validation failure returns failure with stage="validator"
    def test_validation_failure_returns_failure(self) -> None:
        with patch("ingestion.shared_processing.user_pipeline.validate_persisted_artifact") as mock_v:
            mock_v.side_effect = IncompleteArtifactError("aeronautics_manuals", "Checksum mismatch in docstore.json")
            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertFalse(res.success)
            self.assertEqual(res.status, IngestionStatus.FAILED)
            self.assertEqual(res.stage, "validator")

    # 13. Successful pipeline returns correct counts
    def test_successful_pipeline_returns_counts(self) -> None:
        res = run_user_pipeline(
            source=self.sample_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertEqual(res.documents, 2)
        self.assertGreater(res.chunks, 0)
        self.assertEqual(res.embedded, res.chunks)
        self.assertEqual(res.failed_embeddings, 0)

    # 14. Failed pipeline returns the stage that failed
    def test_failed_pipeline_stage_reporting(self) -> None:
        failing_embedder = MockEmbeddingEngine(fail_on_text="Boeing")
        res = run_user_pipeline(
            source=self.sample_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=failing_embedder,
            allow_in_memory=True,
        )
        self.assertEqual(res.stage, "embedder")

    # 15. One ingestion ID is used throughout the request
    def test_single_ingestion_id_propagation(self) -> None:
        custom_id = "user_req_2026_x99"
        res = run_user_pipeline(
            source=self.sample_docs,
            ingestion_id=custom_id,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertEqual(res.ingestion_id, custom_id)
        manifest_path = self.persist_dir / self.collection_type / "manifest.json"
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_data["ingestion_id"], custom_id)
        self.assertEqual(manifest_data["version"], custom_id)

    # 16. Collection type is obtained from configuration, not hardcoded
    def test_dynamic_collection_type(self) -> None:
        for custom_col in ["satellite_systems", "avionics_manuals"]:
            cfg = UserPipelineConfig(
                collection_type=custom_col,
                persist_dir=str(self.persist_dir),
                vector_config=VectorStoreConfig(collection_name=custom_col),
            )
            res = run_user_pipeline(
                source=self.sample_docs,
                config=cfg,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertTrue(res.success)
            self.assertEqual(res.collection_type, custom_col)
            self.assertTrue((self.persist_dir / custom_col / "manifest.json").is_file())

    # 17. Invalid collection type is rejected by validation layer
    def test_invalid_collection_type_rejected(self) -> None:
        invalid_col = "../invalid/path/name"
        res = run_user_pipeline(
            source=self.sample_docs,
            collection_type=invalid_col,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        self.assertIn("Invalid collection type", res.error)

    # 18. Core pipeline does not import or raise HTTPException
    def test_no_http_exception_in_core_pipeline(self) -> None:
        # User pipeline returns domain result or raises PipelineError, never HTTPException
        res = run_user_pipeline(
            source=None,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        self.assertNotIsInstance(res.error, Exception)
        self.assertNotIn("HTTPException", str(type(res)))

    # 19. No system-pipeline retry loop exists in user pipeline
    def test_no_system_retry_loop(self) -> None:
        call_count = 0

        def counting_embed(*args: Any, **kwargs: Any) -> None:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Embedder permanent failure")

        with patch("ingestion.shared_processing.user_pipeline.embed_chunks", side_effect=counting_embed):
            res = run_user_pipeline(
                source=self.sample_docs,
                config=self.config,
                client=self.mock_qdrant,
                embedding_engine=self.mock_embedder,
                allow_in_memory=True,
            )
            self.assertFalse(res.success)
            # Must only attempt once, no retry loop
            self.assertEqual(call_count, 1)

    # 20. No unrelated persisted data in persist/ is deleted during cleanup
    def test_unrelated_persist_data_preserved_on_failure(self) -> None:
        # Create an existing production collection
        prod_col_dir = self.persist_dir / "production_unrelated"
        prod_col_dir.mkdir(parents=True, exist_ok=True)
        prod_manifest = prod_col_dir / "manifest.json"
        prod_manifest.write_text('{"collection_type": "production_unrelated"}', encoding="utf-8")

        # Run a failing user ingestion on a different collection
        failing_embedder = MockEmbeddingEngine(fail_on_text="Boeing")
        res = run_user_pipeline(
            source=self.sample_docs,
            collection_type="failing_collection",
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=failing_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)

        # Verify existing unrelated production collection is completely untouched
        self.assertTrue(prod_manifest.is_file())
        self.assertEqual(prod_manifest.read_text(encoding="utf-8"), '{"collection_type": "production_unrelated"}')

    # 21. No secrets appear in errors or log output
    def test_no_secrets_in_errors(self) -> None:
        res = run_user_pipeline(
            source=Path(self.temp_dir) / "missing_file.pdf",
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertFalse(res.success)
        for secret_token in ["aws_secret_access_key", "qdrant_api_key", "sk-proj-"]:
            self.assertNotIn(secret_token, res.error.lower())

    # 22. User upload from raw bytes (HTTP UploadFile simulation)
    def test_user_upload_from_raw_bytes(self) -> None:
        text_content = b"Turbofan engines produce thrust via a combination of core exhaust and bypass air."
        res = run_user_pipeline(
            source=text_content,
            filename="turbofan_specs.txt",
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        self.assertTrue(res.success)
        self.assertEqual(res.documents, 1)
        self.assertGreater(res.chunks, 0)
        self.assertEqual(res.embedded, res.chunks)
        self.assertTrue(res.persisted)

    # 23. UserPipelineResult.to_dict() serialization
    def test_result_to_dict_serialization(self) -> None:
        res = run_user_pipeline(
            source=self.sample_docs,
            config=self.config,
            client=self.mock_qdrant,
            embedding_engine=self.mock_embedder,
            allow_in_memory=True,
        )
        data = res.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["status"], IngestionStatus.COMPLETED)
        self.assertEqual(data["collection_type"], self.collection_type)
        self.assertIn("counts", data)
        self.assertEqual(data["documents"], 2)


if __name__ == "__main__":
    unittest.main()
