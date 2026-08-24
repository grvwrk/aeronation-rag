"""Focused tests for warm index and query-engine reuse."""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from generate import Generate


class GenerateCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        Generate._index_cache.clear()
        Generate._query_engine_cache.clear()
        self.generator = object.__new__(Generate)
        self.generator._persist = str(Path("persist") / "rag_llm")
        self.generator._collection_name = "rag_llm"
        self.generator._qdrant_client = MagicMock()
        self.generator._async_qdrant_client = MagicMock()
        self.generator._config = {
            "QDRANT_ENABLE_HYBRID": True,
            "FASTEMBED_SPARSE_MODEL": "mock-sparse",
            "RAG_SIMILARITY_CUTOFF": 0.7,
            "RAG_CITATION_CHUNK_SIZE": 128,
            "RAG_CITATION_CHUNK_OVERLAP": 8,
            "RAG_SIMILARITY_TOP_K": 3,
            "RAG_STREAMING": True,
        }
        self.generator._prompts = type(
            "Prompts",
            (),
            {
                "citation_template": "{context_str}",
                "qa_template": "{query_str}",
                "refine_template": "{existing_answer}",
            },
        )()
        self.generator._reranker = MagicMock()

    def test_index_loads_once_for_same_persist_collection(self) -> None:
        loaded_index = MagicMock()
        with patch("generate.QdrantVectorStore"), patch(
            "generate.StorageContext.from_defaults", return_value=MagicMock()
        ), patch("generate.load_index_from_storage", return_value=loaded_index) as load:
            first = self.generator._setup_storage_context("rag_llm")
            second = self.generator._setup_storage_context("rag_llm")

        self.assertIs(first, second)
        load.assert_called_once()

    def test_clear_index_cache_forces_reload(self) -> None:
        loaded_index = MagicMock()
        with patch("generate.QdrantVectorStore"), patch(
            "generate.StorageContext.from_defaults", return_value=MagicMock()
        ), patch("generate.load_index_from_storage", return_value=loaded_index) as load:
            self.generator._setup_storage_context("rag_llm")
            Generate.clear_index_cache("persist", "rag_llm")
            self.generator._setup_storage_context("rag_llm")

        self.assertEqual(load.call_count, 2)

    def test_query_engine_reuses_same_filter_key(self) -> None:
        from llama_index.core import Settings as LlamaSettings

        index = MagicMock()
        engine = MagicMock()
        previous_embed = LlamaSettings._embed_model
        previous_llm = LlamaSettings._llm
        LlamaSettings._embed_model = MagicMock()
        LlamaSettings._llm = MagicMock()
        try:
            with patch.object(Generate, "_query_engine_cache", {}), patch(
                "generate.CitationQueryEngine.from_args", return_value=engine
            ) as create:
                self.generator._init_query_engine(index, [])
                self.generator._init_query_engine(index, [])
        finally:
            LlamaSettings._embed_model = previous_embed
            LlamaSettings._llm = previous_llm

        self.assertIs(self.generator.query_engine, engine)
        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
