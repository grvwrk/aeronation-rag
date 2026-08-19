"""Unit tests for the Embedder stage in the ingestion pipeline.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import DimensionMismatchError, EmbedderError, ModelLoadError, PipelineError
from ingestion.shared_processing.embedder import EmbeddingEngine, embed_chunk, embed_chunks
from models import (
    Chunk,
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingResult,
    FailedChunk,
    SparseEmbeddingData,
    StageCounts,
)


class MockDenseModel:
    """Mock dense embedding model returning deterministic float vectors."""

    def __init__(self, dim: int = 384, fail_on_text: str | None = None) -> None:
        self.dim = dim
        self.fail_on_text = fail_on_text
        self.call_count = 0
        self.batches_received: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.call_count += 1
        self.batches_received.append(texts)
        for t in texts:
            if self.fail_on_text and self.fail_on_text in t:
                raise RuntimeError(f"Simulated dense embedding failure on: {t}")
            # Generate deterministic vector based on text length
            vec = [float((i + len(t)) % 10) / 10.0 for i in range(self.dim)]
            yield vec


class MockSparseModel:
    """Mock sparse embedding model returning deterministic (indices, values) pairs."""

    def __init__(self, fail_on_text: str | None = None) -> None:
        self.fail_on_text = fail_on_text
        self.call_count = 0
        self.batches_received: list[list[str]] = []

    def embed(self, texts: list[str]):
        self.call_count += 1
        self.batches_received.append(texts)
        for t in texts:
            if self.fail_on_text and self.fail_on_text in t:
                raise RuntimeError(f"Simulated sparse embedding failure on: {t}")
            indices = [hash(word) % 10000 for word in t.split()]
            values = [1.0 + (len(word) / 10.0) for word in t.split()]
            # Mock sparse embedding object with indices and values
            mock_emb = MagicMock()
            mock_emb.indices = indices
            mock_emb.values = values
            yield mock_emb


class EmbedderUnitTests(unittest.TestCase):
    def setUp(self):
        self.mock_dense = MockDenseModel(dim=384)
        self.mock_sparse = MockSparseModel()
        self.config = EmbeddingConfig(
            dense_model="mock-dense-model",
            sparse_model="mock-sparse-model",
            batch_size=2,
            expected_dimension=384,
        )
        self.engine = EmbeddingEngine(self.config)
        self.engine._dense_model = self.mock_dense
        self.engine._sparse_model = self.mock_sparse

    # 1. Empty input returns empty result without attempting model loading
    def test_empty_input(self):
        engine_mock = MagicMock(spec=EmbeddingEngine)
        result = embed_chunks([], config=self.config, engine=engine_mock)

        self.assertEqual(result.total_chunks, 0)
        self.assertEqual(result.successful_chunks, 0)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(result.embedded_chunks, [])
        self.assertEqual(result.failed_chunks, [])
        engine_mock.get_dense_model.assert_not_called()
        engine_mock.get_sparse_model.assert_not_called()

    # 2. Single chunk produces dense vector and sparse embedding
    def test_single_chunk_embedding(self):
        chunk = Chunk(
            chunk_id="chunk-abc123",
            text="Lift force balances aircraft weight in steady level flight.",
            source_id="aero-doc-1",
            page_num=5,
            metadata={"file_name": "aerodynamics.pdf", "topic": "flight_mechanics"},
        )
        embedded = embed_chunk(chunk, config=self.config, engine=self.engine)

        self.assertIsInstance(embedded, EmbeddedChunk)
        self.assertEqual(embedded.chunk.chunk_id, "chunk-abc123")
        self.assertEqual(embedded.chunk.text, chunk.text)
        self.assertEqual(len(embedded.dense_embedding), 384)
        self.assertIsNotNone(embedded.sparse_embedding)
        self.assertGreater(len(embedded.sparse_embedding.indices), 0)
        self.assertEqual(len(embedded.sparse_embedding.indices), len(embedded.sparse_embedding.values))
        # Verify metadata preserved
        self.assertEqual(embedded.chunk.metadata["topic"], "flight_mechanics")

    # 3. Multiple chunks and batch processing
    def test_batch_processing_and_batch_size_respected(self):
        chunks = [
            Chunk(chunk_id=f"chunk-{i}", text=f"Aerospace engineering paragraph content number {i}", source_id="src1", page_num=1)
            for i in range(5)
        ]
        # Batch size is 2, so 5 chunks should be processed in 3 batches: [2, 2, 1]
        result = embed_chunks(chunks, config=self.config, engine=self.engine)

        self.assertEqual(result.total_chunks, 5)
        self.assertEqual(result.successful_chunks, 5)
        self.assertEqual(result.failed_count, 0)
        self.assertEqual(len(result.embedded_chunks), 5)
        self.assertEqual(result.batches_total, 3)
        self.assertEqual(result.batches_successful, 3)
        self.assertEqual(result.batches_failed, 0)

        self.assertEqual(self.mock_dense.call_count, 3)
        self.assertEqual(len(self.mock_dense.batches_received[0]), 2)
        self.assertEqual(len(self.mock_dense.batches_received[1]), 2)
        self.assertEqual(len(self.mock_dense.batches_received[2]), 1)

    # 4. Dense and sparse relationship to the correct Chunk
    def test_chunk_embedding_relationship_preservation(self):
        chunks = [
            Chunk(chunk_id="chunk-1", text="First unique content", source_id="doc-A", page_num=10, metadata={"idx": 1}),
            Chunk(chunk_id="chunk-2", text="Second unique content with different words", source_id="doc-B", page_num=20, metadata={"idx": 2}),
        ]
        result = embed_chunks(chunks, config=self.config, engine=self.engine)

        self.assertEqual(len(result.embedded_chunks), 2)
        # Check first chunk
        emb1 = result.embedded_chunks[0]
        self.assertEqual(emb1.chunk.chunk_id, "chunk-1")
        self.assertEqual(emb1.chunk.source_id, "doc-A")
        self.assertEqual(emb1.chunk.page_num, 10)
        self.assertEqual(emb1.chunk.metadata["idx"], 1)

        # Check second chunk
        emb2 = result.embedded_chunks[1]
        self.assertEqual(emb2.chunk.chunk_id, "chunk-2")
        self.assertEqual(emb2.chunk.source_id, "doc-B")
        self.assertEqual(emb2.chunk.page_num, 20)
        self.assertEqual(emb2.chunk.metadata["idx"], 2)

    # 5. Vector dimension validation
    def test_vector_dimension_validation(self):
        # Dense model produces 384 dim, but config expects 768
        mismatch_config = EmbeddingConfig(
            dense_model="mock-dense",
            sparse_model="mock-sparse",
            expected_dimension=768,
        )
        chunk = Chunk(chunk_id="c1", text="Sample text", source_id="s1")

        with self.assertRaises(DimensionMismatchError) as ctx:
            embed_chunks([chunk], config=mismatch_config, engine=self.engine, strict=True)

        self.assertIn("Generated dense vector dimension (384) does not match expected dimension (768)", str(ctx.exception))

    # 6. Model loading failure is explicit
    def test_model_loading_failure(self):
        with patch("fastembed.TextEmbedding", side_effect=ImportError("No ONNX runtime found")):
            engine = EmbeddingEngine(EmbeddingConfig(dense_model="nonexistent-model"))
            with self.assertRaises(ModelLoadError) as ctx:
                engine.get_dense_model()
            self.assertIsInstance(ctx.exception, PipelineError)
            self.assertIn("Failed to load dense embedding model", str(ctx.exception))

    # 7. Batch failure in strict=True raises EmbedderError
    def test_batch_failure_strict_mode(self):
        failing_dense = MockDenseModel(fail_on_text="FAIL_THIS_CHUNK")
        engine = EmbeddingEngine(self.config)
        engine._dense_model = failing_dense
        engine._sparse_model = self.mock_sparse

        chunks = [
            Chunk(chunk_id="c1", text="Normal text 1", source_id="s1"),
            Chunk(chunk_id="c2", text="FAIL_THIS_CHUNK text", source_id="s2"),
        ]

        with self.assertRaises(EmbedderError) as ctx:
            embed_chunks(chunks, config=self.config, engine=engine, strict=True)

        self.assertIsInstance(ctx.exception, PipelineError)

    # 8. Batch failure in strict=False captures failed chunks in result.failed_chunks
    def test_batch_failure_non_strict_mode(self):
        failing_dense = MockDenseModel(fail_on_text="FAIL_THIS_CHUNK")
        engine = EmbeddingEngine(self.config)
        engine._dense_model = failing_dense
        engine._sparse_model = self.mock_sparse

        # 4 chunks, batch_size=2: Batch 1 has c1, c2 (fails due to c2). Batch 2 has c3, c4 (succeeds).
        chunks = [
            Chunk(chunk_id="c1", text="Text 1", source_id="s1", page_num=1),
            Chunk(chunk_id="c2", text="FAIL_THIS_CHUNK", source_id="s2", page_num=2),
            Chunk(chunk_id="c3", text="Text 3", source_id="s3", page_num=3),
            Chunk(chunk_id="c4", text="Text 4", source_id="s4", page_num=4),
        ]

        counts = StageCounts()
        result = embed_chunks(chunks, config=self.config, engine=engine, counts=counts, strict=False)

        self.assertEqual(result.total_chunks, 4)
        self.assertEqual(result.successful_chunks, 2)
        self.assertEqual(result.failed_count, 2)
        self.assertEqual(len(result.embedded_chunks), 2)
        self.assertEqual(len(result.failed_chunks), 2)

        # Verify failed chunks are accurately identifiable
        failed_ids = [fc.chunk.chunk_id for fc in result.failed_chunks]
        self.assertIn("c1", failed_ids)
        self.assertIn("c2", failed_ids)

        # Verify successful chunks
        success_ids = [ec.chunk.chunk_id for ec in result.embedded_chunks]
        self.assertIn("c3", success_ids)
        self.assertIn("c4", success_ids)

        # Verify StageCounts telemetry
        self.assertEqual(counts.embeddings_generated, 2)
        self.assertEqual(counts.chunks_failed, 2)

    # 9. Important failure semantics test: 100 chunks, 97 successful, 3 failed
    def test_partial_failure_semantics_100_chunks_3_failures(self):
        failing_dense = MockDenseModel(dim=384, fail_on_text="BAD_CHUNK")
        config = EmbeddingConfig(batch_size=1, expected_dimension=384)
        engine = EmbeddingEngine(config)
        engine._dense_model = failing_dense
        engine._sparse_model = self.mock_sparse

        chunks = []
        for i in range(100):
            text = f"BAD_CHUNK content {i}" if i in (10, 50, 90) else f"Good chunk content {i}"
            chunks.append(Chunk(chunk_id=f"chunk-{i}", text=text, source_id=f"doc-{i // 10}", page_num=(i % 10) + 1))

        result = embed_chunks(chunks, config=config, engine=engine, strict=False)

        self.assertEqual(result.total_chunks, 100)
        self.assertEqual(result.successful_chunks, 97)
        self.assertEqual(result.failed_count, 3)
        self.assertEqual(len(result.embedded_chunks), 97)
        self.assertEqual(len(result.failed_chunks), 3)

        failed_chunk_ids = {fc.chunk.chunk_id for fc in result.failed_chunks}
        self.assertEqual(failed_chunk_ids, {"chunk-10", "chunk-50", "chunk-90"})

    # 10. Environment variable configuration
    def test_embedding_config_from_env(self):
        with patch.dict(
            os.environ,
            {
                "HF_EMBED": "custom-dense-model",
                "FASTEMBED_SPARSE_MODEL": "custom-sparse-model",
                "EMBEDDING_BATCH_SIZE": "64",
                "EMBEDDING_DIMENSION": "768",
            },
        ):
            config = EmbeddingConfig.from_env()
            self.assertEqual(config.dense_model, "custom-dense-model")
            self.assertEqual(config.sparse_model, "custom-sparse-model")
            self.assertEqual(config.batch_size, 64)
            self.assertEqual(config.expected_dimension, 768)

    # 11. Invalid input types handling
    def test_invalid_chunk_type_raises_error_in_strict_mode(self):
        with self.assertRaises(EmbedderError) as ctx:
            embed_chunks(["not a chunk object"], config=self.config, engine=self.engine, strict=True)  # type: ignore
        self.assertIn("Expected Chunk instance", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
