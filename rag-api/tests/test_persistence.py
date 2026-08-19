"""Unit tests for the Persistence stage in the ingestion pipeline.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import IncompleteArtifactError, ManifestError, PersistenceError
from ingestion.shared_processing.persistence import (
    REQUIRED_PERSISTENCE_FILES,
    build_manifest,
    compute_file_sha256,
    generate_ingestion_id,
    load_manifest,
    persist_collection,
    validate_persisted_artifact,
)
from models import (
    ChunkingConfig,
    EmbeddingConfig,
    PersistenceConfig,
    PersistenceResult,
    StageCounts,
    VectorStoreConfig,
)


class PersistenceUnitTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.vector_config = VectorStoreConfig(
            collection_name="rag_llm_v1",
            dimension=384,
            distance="Cosine",
            enable_hybrid=True,
            dense_vector_name="text-dense",
            sparse_vector_name="text-sparse-new",
        )
        self.embedding_config = EmbeddingConfig(
            dense_model="sentence-transformers/all-MiniLM-L6-v2",
            sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            expected_dimension=384,
        )
        self.chunking_config = ChunkingConfig(
            chunk_size=512,
            chunk_overlap=64,
            min_chunk_size=0,
        )
        self.counts = StageCounts(
            files_seen=3,
            docs_loaded=10,
            docs_cleaned=10,
            docs_discarded=0,
            chunks_created=25,
            embeddings_generated=25,
            chunks_failed=0,
            vectors_inserted=25,
            vectors_failed=0,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Ingestion ID generation
    def test_generate_ingestion_id(self):
        id1 = generate_ingestion_id("rag_llm")
        id2 = generate_ingestion_id("rag_llm")
        self.assertTrue(id1.startswith("rag_llm_"))
        self.assertNotEqual(id1, id2)

    # 2. File SHA-256 computation
    def test_compute_file_sha256(self):
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Aerospace engineering knowledge", encoding="utf-8")
        h1 = compute_file_sha256(test_file)
        self.assertTrue(h1.startswith("sha256:"))
        self.assertEqual(len(h1), 7 + 64)

        with self.assertRaises(FileNotFoundError):
            compute_file_sha256(Path(self.temp_dir) / "nonexistent.txt")

    # 3. Persistence directory creation and standard files preservation
    def test_persist_collection_creates_standard_files_and_manifest(self):
        result = persist_collection(
            collection_type="rag_llm",
            ingestion_id="test_ing_001",
            vector_config=self.vector_config,
            embedding_config=self.embedding_config,
            chunking_config=self.chunking_config,
            counts=self.counts,
            persist_dir=self.temp_dir,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.collection_type, "rag_llm")
        self.assertEqual(result.ingestion_id, "test_ing_001")

        target_dir = Path(self.temp_dir) / "rag_llm"
        self.assertTrue(target_dir.is_dir())

        # Verify all standard LlamaIndex files exist
        for req_file in REQUIRED_PERSISTENCE_FILES:
            file_path = target_dir / req_file
            self.assertTrue(file_path.is_file(), f"Missing required file {req_file}")

        # Verify manifest.json exists
        manifest_path = target_dir / "manifest.json"
        self.assertTrue(manifest_path.is_file())

    # 4. Manifest JSON structure and version binding verification
    def test_manifest_content_and_version_binding(self):
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="version_2026_01",
            vector_config=self.vector_config,
            embedding_config=self.embedding_config,
            chunking_config=self.chunking_config,
            counts=self.counts,
            persist_dir=self.temp_dir,
        )

        manifest = load_manifest(Path(self.temp_dir) / "rag_llm")

        self.assertEqual(manifest["schema_version"], "1.0.0")
        self.assertEqual(manifest["collection_type"], "rag_llm")
        self.assertEqual(manifest["ingestion_id"], "version_2026_01")
        self.assertEqual(manifest["version"], "version_2026_01")

        # Qdrant binding
        self.assertEqual(manifest["qdrant"]["collection_name"], "rag_llm_v1")
        self.assertEqual(manifest["qdrant"]["dimension"], 384)
        self.assertEqual(manifest["qdrant"]["distance"], "Cosine")
        self.assertEqual(manifest["qdrant"]["enable_hybrid"], True)

        # Embedding binding
        self.assertEqual(manifest["embedding"]["dense_model"], "sentence-transformers/all-MiniLM-L6-v2")
        self.assertEqual(manifest["embedding"]["sparse_model"], "Qdrant/bm42-all-minilm-l6-v2-attentions")
        self.assertEqual(manifest["embedding"]["dimension"], 384)

        # Chunking binding
        self.assertEqual(manifest["chunking"]["chunk_size"], 512)
        self.assertEqual(manifest["chunking"]["chunk_overlap"], 64)

        # Counts
        self.assertEqual(manifest["counts"]["docs_loaded"], 10)
        self.assertEqual(manifest["counts"]["chunks_created"], 25)
        self.assertEqual(manifest["counts"]["vectors_inserted"], 25)

        # File inventory & checksums
        files = manifest["persistence"]["files"]
        for req in REQUIRED_PERSISTENCE_FILES:
            self.assertIn(req, files)
        self.assertIn("manifest.json", files)

        checksums = manifest["persistence"]["checksums"]
        for req in REQUIRED_PERSISTENCE_FILES:
            self.assertIn(req, checksums)

    # 5. Strict Zero Secrets guarantee
    def test_no_secrets_in_manifest(self):
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="test_ing_sec",
            vector_config=self.vector_config,
            embedding_config=self.embedding_config,
            counts=self.counts,
            persist_dir=self.temp_dir,
        )

        manifest_file = Path(self.temp_dir) / "rag_llm" / "manifest.json"
        content = manifest_file.read_text(encoding="utf-8")

        for forbidden in ["aws_access_key_id", "aws_secret_access_key", "qdrant_api_key", "api_key"]:
            self.assertNotIn(forbidden, content.lower())

    # 6. Artifact validation succeeds on valid persistence directory
    def test_validate_persisted_artifact_success(self):
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="valid_ing_id",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
        )

        valid_dir = Path(self.temp_dir) / "rag_llm"
        self.assertTrue(validate_persisted_artifact(valid_dir, expected_ingestion_id="valid_ing_id", expected_collection_type="rag_llm"))

    # 7. Missing required persistence file is detected
    def test_missing_required_file_detected(self):
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="valid_ing_id",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
        )

        target_dir = Path(self.temp_dir) / "rag_llm"
        (target_dir / "index_store.json").unlink()

        with self.assertRaises(IncompleteArtifactError) as ctx:
            validate_persisted_artifact(target_dir)
        self.assertIn("missing", str(ctx.exception).lower())

    # 8. Checksum tampering is detected
    def test_checksum_mismatch_detected(self):
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="valid_ing_id",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
        )

        target_dir = Path(self.temp_dir) / "rag_llm"
        # Tamper with docstore.json
        (target_dir / "docstore.json").write_text('{"tampered": true}', encoding="utf-8")

        with self.assertRaises(IncompleteArtifactError) as ctx:
            validate_persisted_artifact(target_dir)
        self.assertIn("Checksum mismatch", str(ctx.exception))

    # 9. Ingestion ID / Version mismatch is detected
    def test_version_mismatch_detected(self):
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="version_A",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
        )

        target_dir = Path(self.temp_dir) / "rag_llm"
        with self.assertRaises(ManifestError) as ctx:
            validate_persisted_artifact(target_dir, expected_ingestion_id="version_B")
        self.assertIn("does not match expected", str(ctx.exception))

    # 10. Atomic staging prevents corruption of existing valid artifact on failure
    def test_atomic_staging_protects_existing_artifact_on_failure(self):
        # 1. Create first valid version
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="initial_v1",
            vector_config=self.vector_config,
            counts=self.counts,
            persist_dir=self.temp_dir,
        )

        initial_manifest = load_manifest(Path(self.temp_dir) / "rag_llm")
        self.assertEqual(initial_manifest["ingestion_id"], "initial_v1")

        # 2. Simulate a failure during second persistence (e.g. failing storage_context.persist)
        failing_storage_context = MagicMock()
        failing_storage_context.persist.side_effect = RuntimeError("Disk full / write error during staging")

        with self.assertRaises(PersistenceError):
            persist_collection(
                collection_type="rag_llm",
                ingestion_id="failing_v2",
                vector_config=self.vector_config,
                persist_dir=self.temp_dir,
                storage_context=failing_storage_context,
            )

        # 3. Verify original artifact is untouched and still valid
        target_dir = Path(self.temp_dir) / "rag_llm"
        self.assertTrue(validate_persisted_artifact(target_dir, expected_ingestion_id="initial_v1"))
        persisted_manifest = load_manifest(target_dir)
        self.assertEqual(persisted_manifest["ingestion_id"], "initial_v1")

    # 11. Re-ingestion creates a new version safely
    def test_re_ingestion_updates_artifact_safely(self):
        # First ingestion
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="v001",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
        )
        self.assertEqual(load_manifest(Path(self.temp_dir) / "rag_llm")["ingestion_id"], "v001")

        # Second ingestion
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="v002",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
        )
        self.assertEqual(load_manifest(Path(self.temp_dir) / "rag_llm")["ingestion_id"], "v002")

    # 12. Roundtrip loading with LlamaIndex StorageContext and load_index_from_storage
    def test_llamaindex_storage_context_roundtrip_loading(self):
        from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        qc = QdrantClient(location=":memory:")
        vs = QdrantVectorStore(client=qc, collection_name="test_roundtrip")
        ctx = StorageContext.from_defaults(vector_store=vs)
        mock_emb = MockEmbedding(embed_dim=384)
        # Create index in storage context
        _ = VectorStoreIndex(nodes=[], storage_context=ctx, embed_model=mock_emb)

        # Persist using persistence stage
        persist_collection(
            collection_type="rag_llm",
            ingestion_id="roundtrip_v1",
            vector_config=self.vector_config,
            persist_dir=self.temp_dir,
            storage_context=ctx,
        )

        target_dir = Path(self.temp_dir) / "rag_llm"
        self.assertTrue(validate_persisted_artifact(target_dir))

        # Test loading via standard LlamaIndex application mechanism (as in generate.py)
        ctx_loaded = StorageContext.from_defaults(persist_dir=str(target_dir), vector_store=vs)
        index_loaded = load_index_from_storage(ctx_loaded, embed_model=mock_emb)
        self.assertIsNotNone(index_loaded)

    # 13. Invalid collection type name is rejected
    def test_invalid_collection_type_rejected(self):
        with self.assertRaises(ValueError):
            persist_collection(collection_type="invalid/path/name", persist_dir=self.temp_dir)


if __name__ == "__main__":
    unittest.main()
