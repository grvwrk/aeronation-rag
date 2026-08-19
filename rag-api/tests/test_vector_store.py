"""Unit tests for the Vector Store stage in the ingestion pipeline.
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import qdrant_client.models as qmodels
from errors import (
    CollectionNotFoundError,
    IncompatibleCollectionError,
    PipelineError,
    VectorDimensionError,
    VectorStoreError,
)
from ingestion.shared_processing.vector_store import (
    build_point_payload,
    create_collection,
    derive_point_id,
    get_collection_info,
    get_qdrant_client,
    sanitize_collection_name,
    upsert_embeddings,
    validate_collection,
)
from models import (
    Chunk,
    EmbeddedChunk,
    FailedVector,
    SparseEmbeddingData,
    StageCounts,
    VectorStoreConfig,
    VectorStoreResult,
)


class MockQdrantClient:
    """Mock QdrantClient simulating collection existence, creation, validation, and upserts."""

    def __init__(self) -> None:
        self.collections: dict[str, Any] = {}
        self.points_store: dict[str, list[qmodels.PointStruct]] = {}
        self.fail_on_upsert = False
        self.recreate_called = False

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def get_collection(self, collection_name: str):
        if collection_name not in self.collections:
            raise RuntimeError(f"Collection {collection_name} not found")
        return self.collections[collection_name]

    def create_collection(
        self,
        collection_name: str,
        vectors_config: Any = None,
        sparse_vectors_config: Any = None,
        **kwargs,
    ):
        mock_info = MagicMock()
        mock_params = MagicMock()
        mock_params.vectors = vectors_config
        mock_params.sparse_vectors = sparse_vectors_config
        mock_info.config.params = mock_params
        mock_info.points_count = len(self.points_store.get(collection_name, []))
        mock_info.vectors_count = mock_info.points_count
        mock_info.status = "green"

        self.collections[collection_name] = mock_info
        if collection_name not in self.points_store:
            self.points_store[collection_name] = []
        return True

    def recreate_collection(
        self,
        collection_name: str,
        vectors_config: Any = None,
        sparse_vectors_config: Any = None,
        **kwargs,
    ):
        self.recreate_called = True
        self.points_store[collection_name] = []
        return self.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            **kwargs,
        )

    def upsert(self, collection_name: str, points: list[qmodels.PointStruct], **kwargs):
        if self.fail_on_upsert:
            raise RuntimeError("Simulated Qdrant cluster connection error")
        if collection_name not in self.points_store:
            self.points_store[collection_name] = []
        self.points_store[collection_name].extend(points)
        if collection_name in self.collections:
            self.collections[collection_name].points_count = len(self.points_store[collection_name])
        return MagicMock()


class VectorStoreUnitTests(unittest.TestCase):
    def setUp(self):
        self.client = MockQdrantClient()
        self.config = VectorStoreConfig(
            collection_name="rag_test",
            dimension=384,
            distance="Cosine",
            enable_hybrid=True,
            dense_vector_name="text-dense",
            sparse_vector_name="text-sparse-new",
            batch_size=2,
        )
        create_collection(self.client, self.config)

    # 1. Collection name validation & sanitization
    def test_sanitize_collection_name(self):
        self.assertEqual(sanitize_collection_name("rag_llm"), "rag_llm")
        self.assertEqual(sanitize_collection_name("rag-docs-2026"), "rag-docs-2026")
        self.assertEqual(sanitize_collection_name("  valid_name  "), "valid_name")

        with self.assertRaises(ValueError):
            sanitize_collection_name("")
        with self.assertRaises(ValueError):
            sanitize_collection_name("invalid collection name with spaces!")
        with self.assertRaises(ValueError):
            sanitize_collection_name("bad/slash")

    # 2. Empty input handling (0 vectors, 0 network requests)
    def test_empty_input_handling(self):
        result = upsert_embeddings(self.client, [], config=self.config)
        self.assertEqual(result.vectors_expected, 0)
        self.assertEqual(result.vectors_inserted, 0)
        self.assertEqual(result.vectors_failed, 0)
        self.assertEqual(result.failed_vectors, [])
        self.assertEqual(len(self.client.points_store["rag_test"]), 0)

    # 3. Deterministic point ID derivation
    def test_derive_point_id(self):
        id1 = derive_point_id("chunk-abc123")
        id2 = derive_point_id("chunk-abc123")
        id_diff = derive_point_id("chunk-xyz987")

        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id_diff)
        self.assertEqual(len(id1), 36)  # Standard UUID string length

        with self.assertRaises(ValueError):
            derive_point_id("")

    # 4. Point payload preservation
    def test_build_point_payload(self):
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Bernoulli equation: p + 1/2 rho v^2 = constant.",
            source_id="physics.pdf",
            page_num=14,
            metadata={"topic": "fluid_dynamics", "section": "3.2", "file_name": "physics.pdf"},
        )
        payload = build_point_payload(chunk)

        self.assertEqual(payload["text"], chunk.text)
        self.assertEqual(payload["source_id"], "physics.pdf")
        self.assertEqual(payload["chunk_id"], "chunk-001")
        self.assertEqual(payload["page_num"], 14)
        self.assertEqual(payload["topic"], "fluid_dynamics")
        self.assertEqual(payload["file_name"], "physics.pdf")
        self.assertIn("metadata", payload)

    # 5. Single chunk upsert with dense and sparse vectors
    def test_single_embedded_chunk_upsert(self):
        chunk = Chunk(
            chunk_id="chunk-001",
            text="Aerodynamics text",
            source_id="src-1",
            page_num=1,
            metadata={"file_name": "aero.pdf"},
        )
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 384,
            sparse_embedding=SparseEmbeddingData(indices=[10, 20, 30], values=[1.5, 2.0, 0.8]),
        )

        counts = StageCounts()
        result = upsert_embeddings(self.client, [embedded], config=self.config, counts=counts)

        self.assertEqual(result.vectors_expected, 1)
        self.assertEqual(result.vectors_inserted, 1)
        self.assertEqual(result.vectors_failed, 0)
        self.assertEqual(counts.vectors_inserted, 1)

        points = self.client.points_store["rag_test"]
        self.assertEqual(len(points), 1)
        pt = points[0]
        self.assertEqual(pt.id, derive_point_id("chunk-001"))
        self.assertIn("text-dense", pt.vector)
        self.assertIn("text-sparse-new", pt.vector)
        self.assertEqual(pt.payload["chunk_id"], "chunk-001")
        self.assertEqual(pt.payload["text"], "Aerodynamics text")

    # 6. Batch upserting respecting batch size
    def test_batch_upserting(self):
        chunks = []
        for i in range(5):
            c = Chunk(chunk_id=f"c-{i}", text=f"Text {i}", source_id="src-batch", page_num=i + 1)
            ec = EmbeddedChunk(
                chunk=c,
                dense_embedding=[0.05 * (i + 1)] * 384,
                sparse_embedding=SparseEmbeddingData(indices=[i, i + 1], values=[1.0, 2.0]),
            )
            chunks.append(ec)

        result = upsert_embeddings(self.client, chunks, config=self.config)
        self.assertEqual(result.vectors_expected, 5)
        self.assertEqual(result.vectors_inserted, 5)
        self.assertEqual(result.vectors_failed, 0)
        self.assertEqual(result.batches_total, 3)  # batch_size=2: [2, 2, 1]
        self.assertEqual(result.batches_successful, 3)
        self.assertEqual(len(self.client.points_store["rag_test"]), 5)

    # 7. Collection compatibility verification & existing collection detection (non-destructive)
    def test_existing_collection_compatibility_and_non_destructive(self):
        # Compatible collection: should return True without error
        self.assertTrue(validate_collection(self.client, self.config))
        self.assertTrue(create_collection(self.client, self.config))
        # Ensure recreate was NOT called
        self.assertFalse(self.client.recreate_called)

    # 8. Incompatible dense dimension raises IncompatibleCollectionError and does NOT recreate
    def test_incompatible_dimension_raises_error(self):
        incompatible_config = VectorStoreConfig(
            collection_name="rag_test",
            dimension=768,  # Collection was created with 384
            distance="Cosine",
        )
        with self.assertRaises(IncompatibleCollectionError) as ctx:
            validate_collection(self.client, incompatible_config)
        self.assertIn("Dense vector dimension mismatch", str(ctx.exception))

        with self.assertRaises(IncompatibleCollectionError):
            create_collection(self.client, incompatible_config)
        self.assertFalse(self.client.recreate_called)

    # 9. Incompatible distance metric raises IncompatibleCollectionError
    def test_incompatible_distance_raises_error(self):
        incompatible_config = VectorStoreConfig(
            collection_name="rag_test",
            dimension=384,
            distance="Dot",  # Collection was created with Cosine
        )
        with self.assertRaises(IncompatibleCollectionError) as ctx:
            validate_collection(self.client, incompatible_config)
        self.assertIn("Distance metric mismatch", str(ctx.exception))

    # 10. Non-existent collection raises CollectionNotFoundError
    def test_nonexistent_collection_validation(self):
        cfg = VectorStoreConfig(collection_name="missing_col")
        with self.assertRaises(CollectionNotFoundError):
            validate_collection(self.client, cfg)

    # 11. Chunk dense vector dimension mismatch raises VectorDimensionError
    def test_vector_dimension_mismatch_on_chunk(self):
        chunk = Chunk(chunk_id="bad-dim", text="Text", source_id="src")
        # 128 dim instead of expected 384
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 128,
            sparse_embedding=SparseEmbeddingData(indices=[1], values=[1.0]),
        )

        with self.assertRaises(VectorDimensionError) as ctx:
            upsert_embeddings(self.client, [embedded], config=self.config, strict=True)
        self.assertIsInstance(ctx.exception, PipelineError)
        self.assertIn("Dense vector dimension mismatch", str(ctx.exception))

    # 12. Missing sparse embedding when hybrid enabled
    def test_missing_sparse_embedding_hybrid_mode(self):
        chunk = Chunk(chunk_id="no-sparse", text="Text", source_id="src")
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 384,
            sparse_embedding=None,  # Missing sparse
        )

        with self.assertRaises(VectorStoreError) as ctx:
            upsert_embeddings(self.client, [embedded], config=self.config, strict=True)
        self.assertIn("Missing or invalid sparse embedding", str(ctx.exception))

    # 13. Sparse vector indices/values length mismatch rejected
    def test_sparse_vector_length_mismatch_rejected(self):
        chunk = Chunk(chunk_id="mismatched-sparse", text="Text", source_id="src")
        # 3 indices, 1 value
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 384,
            sparse_embedding=SparseEmbeddingData(indices=[1, 5, 8], values=[0.2]),
        )

        with self.assertRaises(VectorStoreError) as ctx:
            upsert_embeddings(self.client, [embedded], config=self.config, strict=True)
        self.assertIn("length mismatch", str(ctx.exception))

    # 14. Malformed sparse vector (non-numeric values) rejected
    def test_sparse_vector_non_numeric_rejected(self):
        chunk = Chunk(chunk_id="non-num-sparse", text="Text", source_id="src")
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 384,
            sparse_embedding=SparseEmbeddingData(indices=["bad_idx"], values=["bad_val"]),  # type: ignore
        )

        with self.assertRaises(VectorStoreError) as ctx:
            upsert_embeddings(self.client, [embedded], config=self.config, strict=True)
        self.assertIn("non-numeric", str(ctx.exception))

    # 15. Batch failure in strict=True raises VectorStoreError
    def test_batch_failure_strict_mode(self):
        self.client.fail_on_upsert = True
        chunk = Chunk(chunk_id="c1", text="Text", source_id="s1")
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 384,
            sparse_embedding=SparseEmbeddingData(indices=[1], values=[1.0]),
        )

        with self.assertRaises(VectorStoreError):
            upsert_embeddings(self.client, [embedded], config=self.config, strict=True)

    # 16. Batch failure in strict=False captures failed vectors
    def test_batch_failure_non_strict_mode(self):
        self.client.fail_on_upsert = True
        chunk = Chunk(chunk_id="c1", text="Text", source_id="s1")
        embedded = EmbeddedChunk(
            chunk=chunk,
            dense_embedding=[0.1] * 384,
            sparse_embedding=SparseEmbeddingData(indices=[1], values=[1.0]),
        )

        counts = StageCounts()
        result = upsert_embeddings(self.client, [embedded], config=self.config, counts=counts, strict=False)

        self.assertEqual(result.vectors_expected, 1)
        self.assertEqual(result.vectors_inserted, 0)
        self.assertEqual(result.vectors_failed, 1)
        self.assertEqual(len(result.failed_vectors), 1)
        self.assertEqual(result.failed_vectors[0].chunk_id, "c1")
        self.assertEqual(counts.vectors_failed, 1)

    # 17. Partial failure statistics verification (e.g. 100 vectors, 97 inserted, 3 invalid)
    def test_partial_failure_statistics(self):
        chunks = []
        for i in range(100):
            c = Chunk(chunk_id=f"chunk-{i}", text=f"Text {i}", source_id="src")
            # Invalidate items 10, 50, 90 (wrong dimension)
            dim = 128 if i in (10, 50, 90) else 384
            ec = EmbeddedChunk(
                chunk=c,
                dense_embedding=[0.1] * dim,
                sparse_embedding=SparseEmbeddingData(indices=[i], values=[1.0]),
            )
            chunks.append(ec)

        config_single = VectorStoreConfig(
            collection_name="rag_test",
            dimension=384,
            batch_size=1,
        )

        result = upsert_embeddings(self.client, chunks, config=config_single, strict=False)

        self.assertEqual(result.vectors_expected, 100)
        self.assertEqual(result.vectors_inserted, 97)
        self.assertEqual(result.vectors_failed, 3)
        self.assertEqual(len(result.failed_vectors), 3)
        failed_ids = {fv.chunk_id for fv in result.failed_vectors}
        self.assertEqual(failed_ids, {"chunk-10", "chunk-50", "chunk-90"})

    # 18. Qdrant client configuration validation: missing URL raises error
    def test_get_qdrant_client_missing_url_raises_error(self):
        cfg = VectorStoreConfig(collection_name="rag_test", url=None)
        with self.assertRaises(VectorStoreError) as ctx:
            get_qdrant_client(cfg, allow_in_memory=False)
        self.assertIn("Qdrant URL is not configured", str(ctx.exception))

    # 19. Explicit test-only in-memory Qdrant client
    def test_get_qdrant_client_explicit_in_memory(self):
        cfg = VectorStoreConfig(collection_name="rag_test", url=None)
        client = get_qdrant_client(cfg, allow_in_memory=True)
        self.assertIsNotNone(client)

    # 20. Collection info retrieval
    def test_get_collection_info(self):
        info = get_collection_info(self.client, "rag_test")
        self.assertEqual(info["collection_name"], "rag_test")
        self.assertEqual(info["status"], "green")


if __name__ == "__main__":
    unittest.main()
