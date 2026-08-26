import os
import asyncio
import json
import uuid
import time
import warnings
import logging
import re
import threading
from pathlib import Path
from typing import ClassVar, Optional, Dict, AsyncGenerator, List, Any, Union
from pydantic import BaseModel
from urllib.parse import quote

import boto3
import qdrant_client

# Compatibility shim for newer qdrant-client versions with llama-index-vector-stores-qdrant
try:
    import qdrant_client.qdrant_fastembed as _qf
    if not hasattr(_qf, "IDF_EMBEDDING_MODELS"):
        _qf.IDF_EMBEDDING_MODELS = []
except Exception:
    pass

import nest_asyncio
from tavily import TavilyClient
from botocore.exceptions import ClientError
from llama_index.core import StorageContext, Settings, load_index_from_storage
from llama_index.core.schema import QueryBundle, Node
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.postprocessor.cohere_rerank import CohereRerank
from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.prompts import PromptTemplate
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.llms.openai_like import OpenAILike
from observability import (
    estimate_cost_usd,
    log_latency,
    log_query_token_usage,
    log_retrieval,
    log_token_stream,
)

warnings.filterwarnings("ignore")

# Configure environment
os.environ["TOKENIZERS_PARALLELISM"] = "false"
nest_asyncio.apply()

logger = logging.getLogger(__name__)


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    citation_template: str = "citation_template.prompt"
    qa_template: str = "qa_template.prompt"
    refine_template: str = "refine_template.prompt"
    conv_title_template: str = "conversation_title.prompt"
    related_queries_template: str = "related_queries.prompt"
    system_prompt: str = "system.prompt"
    tavily_template: str = "tavily.prompt"


class ConditionalCohereRerank(CohereRerank):
    """Avoid the Cohere request unless there are enough candidates to reorder."""

    minimum_nodes: ClassVar[int] = 3

    def _postprocess_nodes(self, nodes, query_bundle=None):
        if len(nodes) < self.minimum_nodes:
            logger.info("stage=rerank skipped; retrieved_nodes=%d", len(nodes))
            log_latency("rerank_skipped", 0, retrieved_nodes=len(nodes))
            return nodes
        started = time.perf_counter()
        result = super()._postprocess_nodes(nodes, query_bundle)
        logger.info("stage=rerank duration=%.3fs nodes=%d", time.perf_counter() - started, len(nodes))
        log_latency("rerank", time.perf_counter() - started, retrieved_nodes=len(nodes))
        return result


class StorageManager:
    """Manages S3 storage operations including persist directory and chat history."""

    _s3_client = None
    _chat_hist = None
    _chat_id = None
    _persist_dir_cache = {}

    def __init__(self, config: Dict[str, Any], secret: Dict[str, Any]):
        self._config = config
        self._secret = secret
        self._init_s3_client()

    def _init_s3_client(self) -> boto3.client:
        """Initialize S3 client with credentials."""
        if StorageManager._s3_client is None:
            try:
                logger.info("Initializing S3 client...")
                StorageManager._s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self._secret["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=self._secret["AWS_SECRET_ACCESS_KEY"],
                    region_name=self._config["AWS_REGION"],
                )
            except ClientError as e:
                logger.error(f"Failed to initialize S3 client: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error initializing S3 client: {e}")
                raise

    def load_persist_dir(self, persist_dir: str, collection_name: str) -> str:
        cache_key = f"{persist_dir}/{collection_name}"
        if cache_key in StorageManager._persist_dir_cache:
            logger.info(
                f"Using cached persist directory: {StorageManager._persist_dir_cache[cache_key]}"
            )
            return StorageManager._persist_dir_cache[cache_key]

        try:
            logger.info(f"Loading persist directory for collection: {collection_name}")
            s3_persist_path = f"{persist_dir}/{collection_name}"

            s3_response = StorageManager._s3_client.list_objects_v2(
                Bucket=self._config["S3_PERSIST_BUCKET"], Prefix=s3_persist_path
            )

            if "Contents" not in s3_response:
                logger.error(f"Persist directory not found: {s3_persist_path}")
                raise ValueError(f"Persist directory {s3_persist_path} not found")

            local_persist_dir = Path(
                self._config["S3_PERSIST_DIR"], persist_dir, collection_name
            )
            local_persist_dir.mkdir(parents=True, exist_ok=True)

            for obj in s3_response["Contents"]:
                if obj["Key"].endswith("/"):
                    continue

                file_path = Path(self._config["S3_PERSIST_DIR"], obj["Key"])
                if file_path.exists():
                    logger.info(f"Deleting existing file: {file_path}")
                    file_path.unlink()

                logger.info(f"Downloading file: {file_path}")
                file_path.parent.mkdir(parents=True, exist_ok=True)

                StorageManager._s3_client.download_file(
                    self._config["S3_PERSIST_BUCKET"], obj["Key"], str(file_path)
                )

            logger.info(f"Successfully loaded persist directory to {local_persist_dir}")
            StorageManager._persist_dir_cache[cache_key] = local_persist_dir
            return local_persist_dir

        except ClientError as e:
            logger.error(f"S3 error loading persist directory: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading persist directory: {e}")
            raise

    @classmethod
    def clear_persist_dir_cache(cls, persist_dir: str, collection_name: str) -> None:
        """Forget one downloaded persistence directory after re-ingestion."""
        cls._persist_dir_cache.pop(f"{persist_dir}/{collection_name}", None)

    def load_chat_history(self, curr_chat_id: str) -> Optional[str]:
        """Load chat history for given chat ID."""
        try:
            logger.info(f"Loading chat history for chat ID: {curr_chat_id}")

            if (
                curr_chat_id == StorageManager._chat_id
                and StorageManager._chat_hist is not None
            ):
                logger.debug("Returning cached chat history")
                return StorageManager._chat_hist

            StorageManager._chat_id = curr_chat_id
            chat_summ_file = (
                f"{self._config['S3_CHAT_HISTORY']}/{StorageManager._chat_id}.md"
            )

            s3_chat_summ_obj = StorageManager._s3_client.list_objects_v2(
                Bucket=self._config["S3_LOGS_BUCKET"],
                Delimiter="/",
                Prefix=f"{self._config['S3_CHAT_HISTORY']}/",
            )

            if "Contents" not in s3_chat_summ_obj:
                logger.info("Creating new chat history directory")
                StorageManager._s3_client.put_object(
                    Bucket=self._config["S3_LOGS_BUCKET"],
                    Key=f"{self._config['S3_CHAT_HISTORY']}/",
                )
            elif chat_summ_file in [
                content["Key"] for content in s3_chat_summ_obj["Contents"]
            ]:
                logger.info("Loading existing chat history")
                response = StorageManager._s3_client.get_object(
                    Bucket=self._config["S3_LOGS_BUCKET"], Key=chat_summ_file
                )
                StorageManager._chat_hist = response["Body"].read().decode("utf-8")

        except ClientError as e:
            logger.error(f"S3 error loading chat history: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading chat history: {e}")
            raise

    @property
    def chat_hist(self) -> Optional[str]:
        """Get current chat history."""
        return self._chat_hist

    @chat_hist.setter
    def chat_hist(self, value: str):
        """Set chat history."""
        self._chat_hist = value


class PromptManager:
    _prompts = None

    @staticmethod
    def load_prompts(config: Dict[str, Any]) -> PromptConfig:
        if PromptManager._prompts is None:
            try:
                logger.info("Loading prompt templates...")
                prompts = {}

                for prompt_name, prompt_info in PromptConfig.model_fields.items():
                    with open(
                        Path(config["PROMPT_DIR"], prompt_info.default),
                        "r",
                        encoding="utf-8",
                    ) as f:
                        prompts[prompt_name] = f.read().strip()

                logger.info("Successfully loaded all prompt templates")
                PromptManager._prompts = PromptConfig(**prompts)

            except Exception as e:
                logger.error(f"Error loading prompts: {e}")
                raise

        return PromptManager._prompts


class ModelManager:
    """Manages loading and accessing LLM and embedding models."""

    _embed_model = None
    _llm_model = None

    def __init__(self, config: Dict[str, Any], secret: Dict[str, str]):
        self._config = config
        self._secret = secret
        logger.info("Initializing ModelManager...")

    def _load_embed_model(self) -> FastEmbedEmbedding:
        """Load the lightweight fastembed model."""
        try:
            # We map your config model string cleanly into FastEmbed
            model_name = self._config["HF_EMBED"] 
            max_length = self._config.get("HF_EMBED_MAX_LENGTH", 512)

            from pathlib import Path
            repo_root = Path(__file__).resolve().parent

            abs_cache_dir = str(repo_root / ".fastembed_cache")
            logger.info(f"Loading local cached embedding model from: {abs_cache_dir}")

            logger.info(f"Loading lightweight ONNX embedding model: {model_name}")
            return FastEmbedEmbedding(
                model_name=model_name,
                max_length=max_length,
                cache_dir=abs_cache_dir # Render safe temp directory
            )
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise


    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def _load_llm_model(self) -> Union[OpenAI, Ollama, Anthropic, OpenAILike]:
        """Load the LLM model with retry logic."""
        try:
            model_type = self._config["LLM_MODEL_TYPE"]
            logger.info(f"Loading LLM model: {self._config[model_type]['MODEL_NAME']}")

            if model_type == "OPENAI":
                return OpenAI(
                    model=self._config["OPENAI"]["MODEL_NAME"],
                    api_key=self._secret["OPENAI_API_KEY"],
                    temperature=self._config["OPENAI"]["TEMPERATURE"],
                    top_p=self._config["OPENAI"]["TOP_P"],
                    max_tokens=self._config["OPENAI"]["MAX_TOKENS"],
                    timeout=self._config["OPENAI"]["REQUEST_TIMEOUT"],
                )
            elif model_type == "OLLAMA":
                return Ollama(
                    model=self._config["OLLAMA"]["MODEL_NAME"],
                    temperature=self._config["OLLAMA"]["TEMPERATURE"],
                    top_p=self._config["OLLAMA"]["TOP_P"],
                    request_timeout=self._config["OLLAMA"]["REQUEST_TIMEOUT"],
                    additional_kwargs={
                        "num_ctx": self._config["OLLAMA"]["CTX_LENGTH"],
                        "num_predict": self._config["OLLAMA"]["PREDICT_LENGTH"],
                        "cache": False,
                    },
                )
            elif model_type == "ANTHROPIC":
                return Anthropic(
                    model=self._config["ANTHROPIC"]["MODEL_NAME"],
                    api_key=self._secret["ANTHROPIC_API_KEY"],
                    temperature=self._config["ANTHROPIC"]["TEMPERATURE"],
                    max_tokens=self._config["ANTHROPIC"]["MAX_TOKENS"],
                    timeout=self._config["ANTHROPIC"]["REQUEST_TIMEOUT"],
                )
            elif model_type == "GROQ":
                return OpenAILike(
                    model=self._config["GROQ"]["MODEL_NAME"],
                    api_key=self._secret["GROQ_API_KEY"],
                    additional_kwargs={"reasoning_effort": "low"},
                    # Groq OpenAI-compatible endpoint
                    api_base=self._config["GROQ"]["API_BASE"],
                    is_chat_model=True,
                    temperature=self._config["GROQ"]["TEMPERATURE"],
                    max_tokens=self._config["GROQ"]["MAX_TOKENS"],
                    timeout=self._config["GROQ"]["REQUEST_TIMEOUT"],

                    reuse_client=True,
                )
            else:
                raise ValueError(f"Invalid LLM model type: {model_type}")

        except Exception as e:
            logger.error(f"Error loading LLM model: {e}")
            raise

    @property
    def embed_model(self) -> FastEmbedEmbedding:
        """Get the loaded embedding model."""
        if ModelManager._embed_model is None:
            ModelManager._embed_model = self._load_embed_model()
        return ModelManager._embed_model

    @property
    def llm_model(self) -> Union[OpenAI, Ollama, Anthropic, OpenAILike]:
        """Get the loaded LLM model."""
        if ModelManager._llm_model is None:
            ModelManager._llm_model = self._load_llm_model()
        return ModelManager._llm_model


class Generate:
    """Main class for generating answers to user queries."""

    _index_cache: dict[tuple[str, str], Any] = {}
    _query_engine_cache: dict[tuple[str, str, tuple[tuple[str, str], ...]], Any] = {}
    _index_cache_lock = threading.Lock()

    @classmethod
    def clear_index_cache(cls, persist_dir: str, collection_name: str) -> None:
        """Invalidate a loaded index after its persistence artifact changes."""
        key = (str((Path(persist_dir) / collection_name).resolve()), collection_name)
        cls._index_cache.pop(key, None)
        cls._query_engine_cache = {
            cache_key: engine
            for cache_key, engine in cls._query_engine_cache.items()
            if cache_key[:2] != key
        }

    def __init__(
        self,
        config: Dict[str, Any],
        secret: Dict[str, Any],
        chat_id: str,
        query: str,
        persist_dir: str,
        collection_name: str,
        s3_manager: StorageManager,
        metadata: Dict[str, Any] = {},
        qdrant_client_instance=None,
        async_qdrant_client_instance=None,
        reranker: Optional[CohereRerank] = None,
        request_id: Optional[str] = None,
    ):
        # ... keep all existing __init__ assignments completely the same ...
        self._config = config
        self._secret = secret
        self._query = query
        self._collection_name = collection_name
        self._request_id = request_id or str(uuid.uuid4())
        self._chat_id = chat_id
        self._storage_manager = s3_manager
        self._tavily_client = TavilyClient(api_key=self._secret["TAVILY_API_KEY"])
        self._qdrant_client = qdrant_client_instance
        self._async_qdrant_client = async_qdrant_client_instance
        self._reranker = reranker

        logger.info("Initializing Generate")

        self._prompts = PromptManager.load_prompts(self._config)

        
        # Load chat history and prepare query
        self._storage_manager.load_chat_history(chat_id)
        self._refined_query = self._prepare_query()

        # Setup query engine
        self._persist = self._storage_manager.load_persist_dir(
            persist_dir, collection_name
        )
        index = self._setup_storage_context(collection_name)
        metadata_filters = self._prepare_metadata_filters(metadata)
        self._init_query_engine(index, metadata_filters)

    @classmethod
    async def create(cls, *args, **kwargs) -> "Generate":
        """Build request-specific state off the event loop (S3/index loading is synchronous)."""
        import asyncio
        return await asyncio.to_thread(cls, *args, **kwargs)

    def _setup_storage_context(self, collection_name: str) -> StorageContext:
        """Setup storage context with cached Qdrant vector store."""
        try:
            if self._qdrant_client is None or self._async_qdrant_client is None:
                raise RuntimeError("Shared Qdrant clients must be initialized by AppManager")
            from pathlib import Path
            repo_root = Path(__file__).resolve().parent 
            abs_cache_dir = str(repo_root / ".fastembed_cache")

            # 3. Pass the shared static instances directly into your Vector Store
            cache_key = (str(Path(self._persist).resolve()), collection_name)
            with self._index_cache_lock:
                cached_index = self._index_cache.get(cache_key)
                if cached_index is not None:
                    logger.info("Using cached index for collection: %s", collection_name)
                    return cached_index

                vector_store = QdrantVectorStore(
                    client=self._qdrant_client,
                    aclient=self._async_qdrant_client,
                    collection_name=collection_name,
                    enable_hybrid=self._config["QDRANT_ENABLE_HYBRID"],
                    fastembed_sparse_model=(
                        self._config["FASTEMBED_SPARSE_MODEL"]
                        if self._config["QDRANT_ENABLE_HYBRID"]
                        else None
                    ),
                    dense_vector_name="text-dense",
                    fastembed_cache_dir=abs_cache_dir,
                    prefer_grpc=False,
                    batch_size=16
                )
                storage_context = StorageContext.from_defaults(
                    persist_dir=self._persist, vector_store=vector_store
                )
                index = load_index_from_storage(storage_context)
                self._index_cache[cache_key] = index
                return index
        except Exception as e:
            logger.error(f"Error setting up storage context: {e}")
            raise 
    
    def _prepare_query(self) -> str:
        """Prepare the refined query with chat history."""
        # 8 SPACES INDENTATION FOR THE CODE INSIDE IT
        if self._storage_manager.chat_hist is not None:
            return f"<|CHAT HISTORY|>: {self._storage_manager.chat_hist}\n\n<|QUERY|>: {self._query}"
        return f"<|QUERY|>: {self._query}"



    def _prepare_metadata_filters(
        self, metadata: Dict[str, Any]
    ) -> List[MetadataFilter]:
        """Prepare metadata filters from metadata dict."""
        logger.info(f"Preparing metadata filters: {metadata}")
        return [MetadataFilter(key=key, value=value) for key, value in metadata.items()]

    def _init_query_engine(
        self, index, metadata_filters: Optional[List[MetadataFilter]]
    ) -> None:
        """Initialize the citation query engine."""
        try:
            filter_key = tuple(
                sorted((item.key, str(item.value)) for item in metadata_filters or [])
            )
            cache_key = (
                str(Path(self._persist).resolve()),
                self._collection_name,
                filter_key,
            )
            with self._index_cache_lock:
                cached_engine = self._query_engine_cache.get(cache_key)
                if cached_engine is not None:
                    self.query_engine = cached_engine
                    logger.info(
                        "Using cached query engine for collection: %s",
                        self._collection_name,
                    )
                    return

            sim_processor = SimilarityPostprocessor(
                similarity_cutoff=self._config["RAG_SIMILARITY_CUTOFF"]
            )
            
            if self._reranker is None:
                raise RuntimeError("Shared Cohere reranker must be initialized by AppManager")

            # Access models directly through LlamaIndex's global Settings object
            self.query_engine = CitationQueryEngine.from_args(
                index,
                embed_model=Settings.embed_model,
                chat_mode="context",
                citation_chunk_size=self._config["RAG_CITATION_CHUNK_SIZE"],
                citation_chunk_overlap=self._config["RAG_CITATION_CHUNK_OVERLAP"],
                citation_qa_template=PromptTemplate(
                    self._prompts.citation_template + self._prompts.qa_template
                ),
                citation_refine_template=PromptTemplate(
                    self._prompts.citation_template + self._prompts.refine_template
                ),
                similarity_top_k=self._config["RAG_SIMILARITY_TOP_K"],
                # Filter on Qdrant similarity before Cohere replaces node scores.
                node_postprocessors=[sim_processor, self._reranker],
                filters=MetadataFilters(filters=metadata_filters or []),
                llm=Settings.llm,
                streaming=self._config["RAG_STREAMING"],
            )
            with self._index_cache_lock:
                self._query_engine_cache[cache_key] = self.query_engine
            logger.info("Successfully initialized query engine")

        except Exception as e:
            logger.error(f"Error initializing query engine: {e}")
            raise

    async def generate_answer(self) -> AsyncGenerator[str, None]:
        """Generate and yield the answer for the given query."""
        try:
            answer = ""
            logger.info("Retrieving relevant documents...")
            retrieval_started = time.perf_counter()
            query_bundle = QueryBundle(query_str=self._refined_query)
            retrieved_docs = await self.query_engine.aretrieve(query_bundle)
            logger.info("stage=retrieval duration=%.3fs", time.perf_counter() - retrieval_started)
            log_latency(
                "retrieval", time.perf_counter() - retrieval_started,
                request_id=self._request_id, retrieved_nodes=len(retrieved_docs),
            )
            if len(retrieved_docs) > 3:
                retrieved_docs = retrieved_docs[:3]

            logger.info(f"Retrieved documents: {retrieved_docs}")
            logger.info(
                f"Score of retrieved docs: {[doc.score for doc in retrieved_docs]}"
            )
            for idx, doc in enumerate(retrieved_docs):
                logger.info(f"Document {idx+1}: {doc.node.get_text()}")

            if (
                not retrieved_docs
                or max([doc.score for doc in retrieved_docs])
                <= self._config["RAG_SIMILARITY_CUTOFF"]
            ):
                log_retrieval(
                    request_id=self._request_id,
                    collection_name=self._collection_name,
                    retrieved_nodes=len(retrieved_docs),
                    retrieved_context_ids=[
                        doc.node.metadata.get("chunk_id", doc.node.metadata.get("source_id", "unknown"))
                        for doc in retrieved_docs
                    ],
                    retrieved_scores=[doc.score for doc in retrieved_docs],
                    fallback_used=True,
                )
                logger.warning("No relevant contexts retrieved")
                import asyncio
                search_results = (await asyncio.to_thread(
                    self._tavily_client.search, self._query, max_results=3, search_depth="advanced"
                ))["results"]
                content = "\n\n".join(
                    [
                        f"{idx+1}. Title: {result['title']}\nContent: {result['content']}\nURL: {result['url']}"
                        for idx, result in enumerate(search_results)
                    ]
                )
                tavily_prompt = [
                    ChatMessage(
                        role=MessageRole.SYSTEM,
                        content=self._prompts.system_prompt,
                    ),
                    ChatMessage(
                        role=MessageRole.USER,
                        content=self._prompts.tavily_template.format(
                            search_results=content, query=self._query
                        ),
                    ),
                ]

                tavily_resp = await Settings.llm.astream_chat(
                    tavily_prompt, max_tokens=256
                )

                fallback_started = time.perf_counter()
                fallback_answer = ""
                fallback_tokens = 0
                fallback_first_token_at = None
                async for text in tavily_resp:
                    now = time.perf_counter()
                    if fallback_first_token_at is None:
                        fallback_first_token_at = now
                    fallback_answer += text.delta
                    fallback_tokens += self._estimate_tokens(text.delta)
                    yield json.dumps(
                        {
                            "response_id": str(uuid.uuid4()),
                            "request_id": self._request_id,
                            "type": "tokens",
                            "answer_source": "web_fallback",
                            "text": text.delta,
                        }
                    )

                fallback_duration = time.perf_counter() - fallback_started
                fallback_input_tokens = self._estimate_tokens(self._refined_query)
                log_query_token_usage(
                    request_id=self._request_id,
                    chat_id=self._chat_id,
                    collection_name=self._collection_name,
                    input_tokens_estimate=fallback_input_tokens,
                    output_tokens_estimate=fallback_tokens,
                    total_tokens_estimate=fallback_input_tokens + fallback_tokens,
                    generation_duration_ms=round(fallback_duration * 1000, 2),
                    output_tokens_per_second=round(fallback_tokens / fallback_duration, 2)
                    if fallback_duration
                    else 0,
                    time_to_first_token_ms=round(
                        (fallback_first_token_at - fallback_started) * 1000, 2
                    )
                    if fallback_first_token_at
                    else None,
                    model_name=self._config.get(
                        self._config.get("LLM_MODEL_TYPE", ""), {}
                    ).get("MODEL_NAME"),
                    cost_usd=estimate_cost_usd(
                        fallback_input_tokens, fallback_tokens, self._config
                    ),
                )

                return

            log_retrieval(
                request_id=self._request_id,
                collection_name=self._collection_name,
                retrieved_nodes=len(retrieved_docs),
                reranked_nodes=len(retrieved_docs),
                retrieved_context_ids=[
                    doc.node.metadata.get("chunk_id", doc.node.metadata.get("source_id", "unknown"))
                    for doc in retrieved_docs
                ],
                retrieved_scores=[doc.score for doc in retrieved_docs],
                fallback_used=False,
            )

            # Generate response
            logger.info("Generating response...")
            start_response = time.perf_counter()
            logger.info("stage=generation_start")
            log_latency("generation_start", 0, request_id=self._request_id)
            response = await self.query_engine.aquery(self._refined_query)
            logger.info("stage=generation_stream_ready duration=%.3fs", time.perf_counter() - start_response)
            log_latency("generation_stream_ready", time.perf_counter() - start_response, request_id=self._request_id)

            output_token_estimate = 0
            chunk_sequence = 0
            first_token_at = None
            previous_chunk_at = start_response
            async for text in response.async_response_gen():
                if text != "Empty Response":
                    now = time.perf_counter()
                    if first_token_at is None:
                        first_token_at = now
                        log_latency("time_to_first_token", now - start_response, request_id=self._request_id)
                    answer += text
                    chunk_tokens = self._estimate_tokens(text)
                    output_token_estimate += chunk_tokens
                    chunk_sequence += 1
                    log_token_stream(
                        request_id=self._request_id,
                        chat_id=self._chat_id,
                        sequence=chunk_sequence,
                        elapsed_ms=round((now - start_response) * 1000, 2),
                        inter_chunk_ms=round((now - previous_chunk_at) * 1000, 2),
                        chunk_tokens_estimate=chunk_tokens,
                        cumulative_output_tokens_estimate=output_token_estimate,
                    )
                    previous_chunk_at = now
                    yield json.dumps(
                        {
                            "response_id": str(uuid.uuid4()),
                            "request_id": self._request_id,
                            "type": "tokens",
                            "answer_source": "indexed_corpus",
                            "text": text,
                        }
                    )
            logger.info("stage=generation_end duration=%.3fs", time.perf_counter() - start_response)
            log_latency("generation_end", time.perf_counter() - start_response, request_id=self._request_id)
            generation_duration = time.perf_counter() - start_response
            input_tokens_estimate = self._estimate_tokens(self._refined_query)
            log_query_token_usage(
                request_id=self._request_id,
                chat_id=self._chat_id,
                collection_name=self._collection_name,
                input_tokens_estimate=input_tokens_estimate,
                output_tokens_estimate=output_token_estimate,
                total_tokens_estimate=input_tokens_estimate + output_token_estimate,
                generation_duration_ms=round(generation_duration * 1000, 2),
                output_tokens_per_second=round(output_token_estimate / generation_duration, 2) if generation_duration else 0,
                time_to_first_token_ms=round((first_token_at - start_response) * 1000, 2) if first_token_at else None,
                model_name=self._config.get(self._config.get("LLM_MODEL_TYPE", ""), {}).get("MODEL_NAME"),
                cost_usd=estimate_cost_usd(
                    input_tokens_estimate,
                    output_token_estimate,
                    self._config,
                ),
            )

            # Process contexts and citations
            logger.info("Processing contexts and citations...")
            contexts, answer = self._process_contexts(
                answer, response.source_nodes, retrieved_docs
            )
            yield json.dumps(
                {"response_id": str(uuid.uuid4()), "request_id": self._request_id, "type": "answer", "answer_source": "indexed_corpus", "text": answer}
            )

            yield json.dumps(
                {
                    "response_id": str(uuid.uuid4()),
                    "request_id": self._request_id,
                    "type": "context",
                    "answer_source": "indexed_corpus",
                    "text": json.dumps(contexts),
                }
            )

            # Update chat history
            had_chat_history = self._storage_manager.chat_hist is not None
            self._storage_manager.chat_hist = f"{self._refined_query}\n{answer}\n\n"
            # These values are not needed to complete the answer stream.  Run the
            # independent Groq calls after it has been sent to the client.
            import asyncio
            asyncio.create_task(
                self._generate_follow_up_metadata(response.source_nodes, answer, had_chat_history)
            )
            logger.info("Successfully completed response generation")

        except Exception as e:
            logger.critical(f"Error generating answer: {e}")
            raise

    def _process_contexts(
        self, answer: str, source_nodes: List[Node], retrieved_docs: List[Node]
    ) -> Dict[str, Dict[str, Any]]:
        """Process and format context information from retrieved documents."""
        try:
            logger.debug("Processing context information...")
            extract_pattern = r"^Source \d+:\s*\n"
            cited_nums = re.findall(r"\[(\d+)\]", answer)
            source_lst = []

            for source in source_nodes:
                source_text = re.sub(
                    extract_pattern, "", source.node.get_text(), flags=re.MULTILINE
                ).strip()
                source_lst.append(source_text)

            contexts = {}
            retrieved_counter = 0

            for idx, doc in enumerate(retrieved_docs):
                if str(idx + 1) not in cited_nums:
                    continue

                if doc.text.strip() in source_lst:
                    retrieved_counter += 1
                    contexts[str(retrieved_counter)] = {
                        "file_name": doc.metadata.get("file_name", "unknown"),
                        "page_num": doc.metadata.get("page_num", "unknown"),
                        "chunk": doc.metadata.get("highlighted_chunk") or source_text,
                    }
                    answer = answer.replace(
                        f"[{str(idx+1)}]",
                        f'[[{retrieved_counter}]]({self._config["PDF_BASE_URL"]}{quote(doc.metadata.get("file_name", "unknown"))})',
                    )

            logger.debug(f"Processed {retrieved_counter} context citations")
            return contexts, answer

        except Exception as e:
            logger.error(f"Error processing contexts: {e}")
            raise

    async def _generate_follow_up_metadata(
        self, source_nodes: List[Node], answer: str, had_chat_history: bool
    ) -> None:
        """Generate optional UI metadata without extending response latency."""
        started = time.perf_counter()
        related_prompt = self._prompts.related_queries_template.format(
            query=self._query,
            sources="\n\n".join(doc.node.get_text() for doc in source_nodes),
            answer=answer,
        )
        calls = [Settings.llm.acomplete(related_prompt, max_tokens=128)]
        if not had_chat_history:
            calls.append(Settings.llm.acomplete(
                self._prompts.conv_title_template.format(query=self._query), max_tokens=64
            ))
        try:
            results = await asyncio.gather(*calls)
            logger.info(
                "stage=follow_up_metadata duration=%.3fs related=%s title=%s",
                time.perf_counter() - started,
                results[0].text.strip(),
                results[1].text.strip() if len(results) > 1 else "not-requested",
            )
        except Exception:
            logger.exception("Background related-query/title generation failed")

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate tokens for local telemetry; provider billing remains authoritative."""
        try:
            import tiktoken
            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except Exception:
            return len(text.split())
