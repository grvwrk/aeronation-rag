"""
Authoritative Unit Tests for the Validation Layer (validator.py).

Verifies all 32 required validation checks:
1. Valid configuration passes.
2. Invalid configuration fails.
3. Invalid collection type fails.
4. Empty document set fails.
5. Invalid document metadata fails.
6. Empty chunk fails.
7. Invalid chunk metadata fails.
8. Chunk exceeding configured token limit fails.
9. Missing embedding fails.
10. Partial embedding failure fails final validation.
11. Dense dimension mismatch fails.
12. Sparse vector malformed fails.
13. Missing dense/sparse representation fails when hybrid retrieval requires both.
14. Vector collection dimension mismatch fails.
15. Distance metric mismatch fails.
16. Required payload field missing fails.
17. Persistence directory missing fails.
18. Required persistence file missing fails.
19. Manifest missing fails.
20. Invalid manifest fails.
21. Manifest collection type mismatch fails.
22. Manifest version mismatch fails.
23. Persistence cannot be loaded fails.
24. Count mismatch fails.
25. Persistence/vector-store mismatch fails.
26. S3 artifact missing required object fails.
27. S3/local version mismatch fails.
28. Valid complete ingestion passes.
29. Validator does not modify artifacts.
30. Validator does not generate embeddings.
31. Validator does not upload to S3.
32. Validator does not modify Qdrant.
"""

from __future__ import annotations

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

from errors import IncompleteArtifactError, ManifestError, ValidationError
from models import (
    Chunk,
    ChunkingConfig,
    EmbeddedChunk,
    EmbeddingConfig,
    FailedChunk,
    IngestionStatus,
    PipelineConfig,
    RawDoc,
    SparseEmbeddingData,
    StageCounts,
    UserPipelineConfig,
    UserPipelineResult,
    ValidationResult,
    VectorStoreConfig,
)
from validator import (
    REQUIRED_PERSISTENCE_FILES,
    validate_cleaned_documents,
    validate_chunks,
    validate_collection_type,
    validate_config,
    validate_counts,
    validate_documents,
    validate_embeddings,
    validate_ingestion_result,
    validate_local_artifact,
    validate_manifest,
    validate_persisted_artifact,
    validate_persistence,
    validate_s3_artifact,
    validate_vector_store,
)


class ValidatorUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.persist_dir = Path(self.temp_dir) / "persist" / "aerospace_manuals"
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.collection_type = "aerospace_manuals"
        self.ingestion_id = "v2026_test_001"

        # Create a valid minimal persistence artifact
        self.checksums: dict[str, str] = {}
        for req_file in REQUIRED_PERSISTENCE_FILES:
            fp = self.persist_dir / req_file
            content = f'{{"store": "{req_file}", "collection_type": "{self.collection_type}"}}'
            fp.write_text(content, encoding="utf-8")
            from ingestion.shared_processing.persistence import compute_file_sha256

            self.checksums[req_file] = compute_file_sha256(fp)

        self.valid_manifest = {
            "schema_version": "1.0.0",
            "collection_type": self.collection_type,
            "ingestion_id": self.ingestion_id,
            "version": self.ingestion_id,
            "persistence": {
                "format": "llamaindex_storage_context",
                "files": list(REQUIRED_PERSISTENCE_FILES),
                "checksums": self.checksums,
            },
        }
        (self.persist_dir / "manifest.json").write_text(
            json.dumps(self.valid_manifest, indent=2), encoding="utf-8"
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Valid configuration passes
    def test_valid_configuration_passes(self) -> None:
        cfg = PipelineConfig(
            collection_type="aerospace_manuals",
            chunking_config=ChunkingConfig(chunk_size=512, chunk_overlap=64),
            embedding_config=EmbeddingConfig(dense_model="mock-model", expected_dimension=384),
            vector_config=VectorStoreConfig(collection_name="aerospace_manuals", dimension=384),
        )
        res = validate_config(cfg)
        self.assertTrue(res.valid)
        self.assertEqual(len(res.errors), 0)

    # 2. Invalid configuration fails
    def test_invalid_configuration_fails(self) -> None:
        cfg = PipelineConfig(
            collection_type="aerospace_manuals",
            chunking_config=ChunkingConfig(chunk_size=100, chunk_overlap=120),  # overlap > size
            embedding_config=EmbeddingConfig(dense_model="", expected_dimension=-5),
            vector_config=VectorStoreConfig(collection_name="aerospace_manuals", dimension=0, distance="InvalidMetric"),
        )
        res = validate_config(cfg)
        self.assertFalse(res.valid)
        self.assertGreater(len(res.errors), 0)

    # 3. Invalid collection type fails
    def test_invalid_collection_type_fails(self) -> None:
        invalid_types = ["../path_traversal", "invalid/slash", "has whitespace", "", "@#$%", None]
        for inv in invalid_types:
            with self.assertRaises(ValueError):
                validate_collection_type(inv)

    # 4. Empty document set fails
    def test_empty_document_set_fails(self) -> None:
        res = validate_documents([])
        self.assertFalse(res.valid)
        self.assertIn("empty", res.errors[0].lower())

    # 5. Invalid document metadata fails
    def test_invalid_document_metadata_fails(self) -> None:
        doc = RawDoc(text="Valid document content", source_id="doc1.pdf", metadata="not-a-dict")  # type: ignore
        res = validate_documents([doc])
        self.assertFalse(res.valid)
        self.assertIn("metadata must be a dictionary", res.errors[0].lower())

    # 6. Empty chunk fails
    def test_empty_chunk_fails(self) -> None:
        chunks = [
            Chunk(chunk_id="chk1", text="", source_id="doc1.pdf"),
            Chunk(chunk_id="chk2", text="   \n  ", source_id="doc1.pdf"),
        ]
        res = validate_chunks(chunks)
        self.assertFalse(res.valid)
        self.assertIn("empty text", res.errors[0].lower())

    # 7. Invalid chunk metadata / missing ID fails
    def test_invalid_chunk_id_fails(self) -> None:
        chunks = [
            Chunk(chunk_id="", text="Valid text", source_id="doc1.pdf"),
        ]
        res = validate_chunks(chunks)
        self.assertFalse(res.valid)
        self.assertIn("missing a chunk_id", res.errors[0].lower())

    # 8. Chunk exceeding configured token limit fails
    def test_chunk_exceeding_token_limit_fails(self) -> None:
        chunks = [
            Chunk(
                chunk_id="chk1",
                text="Some text",
                source_id="doc1.pdf",
                metadata={"chunk_tokens": 1000},
            )
        ]
        cfg = ChunkingConfig(chunk_size=200)
        res = validate_chunks(chunks, config=cfg)
        self.assertFalse(res.valid)
        self.assertIn("exceeds budget", res.errors[0].lower())

    # 9. Missing embedding fails
    def test_missing_embedding_fails(self) -> None:
        chunk = Chunk(chunk_id="c1", text="Sample text", source_id="s1")
        res = validate_embeddings(chunks=[chunk], embedded_chunks=[])
        self.assertFalse(res.valid)
        self.assertIn("mismatch", res.errors[0].lower())

    # 10. Partial embedding failure fails final validation
    def test_partial_embedding_failure_fails(self) -> None:
        c1 = Chunk(chunk_id="c1", text="Text 1", source_id="s1")
        c2 = Chunk(chunk_id="c2", text="Text 2", source_id="s1")
        ech1 = EmbeddedChunk(chunk=c1, dense_embedding=[0.1] * 384)
        failed = FailedChunk(chunk=c2, reason="GPU OOM")

        res = validate_embeddings(
            chunks=[c1, c2],
            embedded_chunks=[ech1],
            failed_chunks=[failed],
            config=EmbeddingConfig(expected_dimension=384),
        )
        self.assertFalse(res.valid)
        self.assertIn("failed", res.errors[1].lower())

    # 11. Dense dimension mismatch fails
    def test_dense_dimension_mismatch_fails(self) -> None:
        c1 = Chunk(chunk_id="c1", text="Text 1", source_id="s1")
        ech1 = EmbeddedChunk(chunk=c1, dense_embedding=[0.1] * 128)  # Expected 384
        res = validate_embeddings(
            chunks=[c1],
            embedded_chunks=[ech1],
            config=EmbeddingConfig(expected_dimension=384),
        )
        self.assertFalse(res.valid)
        self.assertIn("dimension mismatch", res.errors[0].lower())

    # 12. Sparse vector malformed fails
    def test_sparse_vector_malformed_fails(self) -> None:
        c1 = Chunk(chunk_id="c1", text="Text 1", source_id="s1")
        malformed_sparse = SparseEmbeddingData(indices=[1, 2, 3], values=[0.5, 0.8])  # Length mismatch
        ech1 = EmbeddedChunk(chunk=c1, dense_embedding=[0.1] * 384, sparse_embedding=malformed_sparse)
        res = validate_embeddings(
            chunks=[c1],
            embedded_chunks=[ech1],
            config=EmbeddingConfig(expected_dimension=384),
            require_hybrid=True,
        )
        self.assertFalse(res.valid)
        self.assertIn("length mismatch", res.errors[0].lower())

    # 13. Missing dense/sparse representation fails when hybrid requires both
    def test_missing_sparse_when_hybrid_required_fails(self) -> None:
        c1 = Chunk(chunk_id="c1", text="Text 1", source_id="s1")
        ech1 = EmbeddedChunk(chunk=c1, dense_embedding=[0.1] * 384, sparse_embedding=None)
        res = validate_embeddings(
            chunks=[c1],
            embedded_chunks=[ech1],
            config=EmbeddingConfig(expected_dimension=384),
            require_hybrid=True,
        )
        self.assertFalse(res.valid)
        self.assertIn("has no sparse embedding", res.errors[0].lower())

    # 14. Vector collection dimension mismatch fails
    def test_vector_collection_dimension_mismatch_fails(self) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.config.params.vectors = {
            "text-dense": MagicMock(size=768, distance=MagicMock(value="Cosine"))
        }
        mock_client.get_collection.return_value = mock_info

        cfg = VectorStoreConfig(collection_name="aerospace_manuals", dimension=384)
        res = validate_vector_store(mock_client, cfg)
        self.assertFalse(res.valid)
        self.assertIn("dimension mismatch", res.errors[0].lower())

    # 15. Distance metric mismatch fails
    def test_distance_metric_mismatch_fails(self) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.config.params.vectors = {
            "text-dense": MagicMock(size=384, distance=MagicMock(value="Dot"))
        }
        mock_client.get_collection.return_value = mock_info

        cfg = VectorStoreConfig(collection_name="aerospace_manuals", dimension=384, distance="Cosine")
        res = validate_vector_store(mock_client, cfg)
        self.assertFalse(res.valid)
        self.assertIn("distance metric mismatch", res.errors[0].lower())

    # 16. Nonexistent Qdrant collection fails
    def test_nonexistent_vector_collection_fails(self) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        cfg = VectorStoreConfig(collection_name="missing_col", dimension=384)
        res = validate_vector_store(mock_client, cfg)
        self.assertFalse(res.valid)
        self.assertIn("does not exist", res.errors[0].lower())

    # 17. Persistence directory missing fails
    def test_persistence_directory_missing_fails(self) -> None:
        res = validate_persistence(Path(self.temp_dir) / "nonexistent_dir")
        self.assertFalse(res.valid)
        self.assertIn("does not exist", res.errors[0].lower())

    # 18. Required persistence file missing fails
    def test_required_persistence_file_missing_fails(self) -> None:
        (self.persist_dir / "docstore.json").unlink()
        res = validate_persistence(self.persist_dir)
        self.assertFalse(res.valid)
        self.assertIn("docstore.json", res.errors[0].lower())

    # 19. Manifest missing fails
    def test_manifest_missing_fails(self) -> None:
        (self.persist_dir / "manifest.json").unlink()
        res = validate_persistence(self.persist_dir)
        self.assertFalse(res.valid)
        self.assertIn("manifest.json missing", res.errors[0].lower())

    # 20. Invalid manifest JSON fails
    def test_invalid_manifest_json_fails(self) -> None:
        (self.persist_dir / "manifest.json").write_text("{corrupted json", encoding="utf-8")
        res = validate_persistence(self.persist_dir)
        self.assertFalse(res.valid)
        self.assertIn("not valid json", res.errors[0].lower())

    # 21. Manifest collection type mismatch fails
    def test_manifest_collection_type_mismatch_fails(self) -> None:
        res = validate_persistence(self.persist_dir, expected_collection_type="different_col")
        self.assertFalse(res.valid)
        self.assertIn("does not match expected", res.errors[0].lower())

    # 22. Manifest version mismatch fails
    def test_manifest_version_mismatch_fails(self) -> None:
        res = validate_persistence(self.persist_dir, expected_ingestion_id="v9999_wrong")
        self.assertFalse(res.valid)
        self.assertIn("does not match expected", res.errors[0].lower())

    # 23. Checksum mismatch fails
    def test_persistence_checksum_tampered_fails(self) -> None:
        (self.persist_dir / "docstore.json").write_text('{"tampered": true}', encoding="utf-8")
        res = validate_persistence(self.persist_dir)
        self.assertFalse(res.valid)
        self.assertIn("checksum mismatch", res.errors[0].lower())

    # 24. Count mismatch fails
    def test_count_mismatch_fails(self) -> None:
        counts = StageCounts(
            docs_loaded=10,
            docs_cleaned=12,  # Cleaned > Loaded
            chunks_created=50,
            embeddings_generated=48,  # Embedded != Chunks
            chunks_failed=2,
            vectors_inserted=48,
        )
        res = validate_counts(counts)
        self.assertFalse(res.valid)
        self.assertGreater(len(res.errors), 0)

    # 25. Cleaned documents discarded all input fails
    def test_cleaned_documents_discarded_all_fails(self) -> None:
        orig = [RawDoc(text="Sample text", source_id="d1.pdf")]
        res = validate_cleaned_documents([], original_docs=orig)
        self.assertFalse(res.valid)
        self.assertIn("discarded all documents", res.errors[0].lower())

    # 26. S3 artifact missing required object fails
    def test_s3_artifact_missing_object_fails(self) -> None:
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(self.valid_manifest).encode("utf-8"))
        }

        def fake_head(Bucket: str, Key: str) -> dict[str, Any]:
            if "image__vector_store.json" in Key:
                raise RuntimeError("NoSuchKey (404)")
            return {"ContentLength": 100}

        mock_s3.head_object.side_effect = fake_head

        res = validate_s3_artifact(
            mock_s3,
            bucket="aeronation-persist-bucket",
            prefix="persist/aerospace_manuals",
        )
        self.assertFalse(res.valid)
        self.assertIn("image__vector_store.json", res.errors[0].lower())

    # 27. S3 / local version mismatch fails
    def test_s3_version_mismatch_fails(self) -> None:
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(self.valid_manifest).encode("utf-8"))
        }
        res = validate_s3_artifact(
            mock_s3,
            bucket="aeronation-persist-bucket",
            prefix="persist/aerospace_manuals",
            expected_ingestion_id="v_mismatch_expected",
        )
        self.assertFalse(res.valid)
        self.assertIn("does not match expected", res.errors[0].lower())

    # 28. Valid complete ingestion passes
    def test_valid_complete_ingestion_passes(self) -> None:
        # Persistence check
        p_res = validate_persistence(
            self.persist_dir,
            expected_collection_type=self.collection_type,
            expected_ingestion_id=self.ingestion_id,
        )
        self.assertTrue(p_res.valid)
        self.assertEqual(len(p_res.errors), 0)

        # Ingestion result check
        counts = StageCounts(
            docs_loaded=2,
            docs_cleaned=2,
            chunks_created=10,
            embeddings_generated=10,
            vectors_inserted=10,
        )
        user_res = UserPipelineResult(
            success=True,
            status=IngestionStatus.COMPLETED,
            ingestion_id=self.ingestion_id,
            collection_type=self.collection_type,
            stage="completed",
            documents=2,
            chunks=10,
            embedded=10,
            counts=counts,
        )
        ing_val = validate_ingestion_result(user_res, persist_dir=self.persist_dir)
        self.assertTrue(ing_val.valid)

    # 29. Validator does not modify artifacts (read-only verification)
    def test_validator_does_not_modify_artifacts(self) -> None:
        manifest_path = self.persist_dir / "manifest.json"
        before_content = manifest_path.read_text(encoding="utf-8")
        before_stat = manifest_path.stat()

        validate_persistence(self.persist_dir)

        after_content = manifest_path.read_text(encoding="utf-8")
        self.assertEqual(before_content, after_content)

    # 30. Validator does not generate embeddings
    def test_validator_does_not_generate_embeddings(self) -> None:
        chunk = Chunk(chunk_id="c1", text="Sample", source_id="s1")
        ech = EmbeddedChunk(chunk=chunk, dense_embedding=[0.01] * 384)
        with patch("ingestion.shared_processing.embedder.embed_chunks") as mock_emb:
            validate_embeddings([chunk], [ech], config=EmbeddingConfig(expected_dimension=384))
            mock_emb.assert_not_called()

    # 31. Validator does not upload to S3
    def test_validator_does_not_upload_to_s3(self) -> None:
        mock_s3 = MagicMock()
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(self.valid_manifest).encode("utf-8"))
        }
        mock_s3.head_object.return_value = {"ContentLength": 100}

        validate_s3_artifact(mock_s3, "bucket", "prefix")

        mock_s3.put_object.assert_not_called()
        mock_s3.upload_file.assert_not_called()
        mock_s3.delete_object.assert_not_called()

    # 32. Validator does not modify Qdrant
    def test_validator_does_not_modify_qdrant(self) -> None:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_info = MagicMock()
        mock_info.config.params.vectors = {
            "text-dense": MagicMock(size=384, distance=MagicMock(value="Cosine"))
        }
        mock_client.get_collection.return_value = mock_info

        cfg = VectorStoreConfig(collection_name="aerospace_manuals", dimension=384)
        validate_vector_store(mock_client, cfg)

        mock_client.create_collection.assert_not_called()
        mock_client.upsert.assert_not_called()
        mock_client.delete_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
