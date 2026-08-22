import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["FASTEMBED_CACHE_DIR"] = "./.fastembed_cache"

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
import asyncio
import boto3
import logging
import time
import yaml
import atexit
import socket
import gc
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from botocore.exceptions import ClientError
import watchtower
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai_like import OpenAILike
from tenacity import retry, stop_after_attempt, wait_exponential
from generate import ConditionalCohereRerank, Generate, StorageManager
from observability import configure_local_logging, log_latency
import qdrant_client
from secrets_manager import get_secret

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


# Constants
CONFIG_PATH = Path("config/config.yaml")

logger = logging.getLogger(__name__)

start_time = time.perf_counter()


class PromptConfig(BaseModel):
    """Configuration for prompt templates."""

    profanity_filter: str = "profanity_filter.prompt"
    history_summarizer: str = "history_summarizer.prompt"
    rephrased_query: str = "rephrased_query.prompt"


class RAG(BaseModel):
    """Data model for RAG API requests with validation."""

    chat_id: str = Field(default="test", description="Chat ID")
    query: str = Field(..., description="User query")
    file_name: Optional[str] = Field(default="", description="File name")
    collection_name: Optional[str] = Field(
        default="rag_llm", description="Collection name"
    )
    persist_dir: Optional[str] = Field(
        default="persist", description="Persistent directory"
    )

    @field_validator("chat_id")
    def validate_chat_id(cls, value):
        """Validate chat_id can contain alphanumeric characters, hyphens, and underscores."""
        if not all(c.isalnum() or c in "-_" for c in value):
            logger.warning(f"Invalid chat_id format: {value}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid chat_id. Chat ID must be alphanumeric and can include hyphens and underscores.",
            )
        return value

    @field_validator("query")
    def validate_query(cls, value):
        """Validate request-only invariants; network checks belong in the route."""
        if not value.strip():
            logger.warning("Query is empty")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Query cannot be empty."
            )
        return value



class Settings:
    """Application configuration manager loading from YAML."""

    def __init__(self):
        logger.info("Initializing application settings")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.secret = get_secret(self.config)
        logger.info("Application settings initialized successfully")


class PromptManager:
    """Manages loading and accessing prompt templates."""

    @staticmethod
    def load_prompts(config: Dict[str, Any]) -> PromptConfig:
        """Load all prompt templates from the prompts directory."""
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
            return PromptConfig(**prompts)

        except Exception as e:
            logger.error(f"Error loading prompts: {e}")
            raise


class LogManager:
    """CloudWatch logging configuration manager."""

    @staticmethod
    def setup_logging(config: dict, secret: dict) -> None:
        """Initialize CloudWatch logging with instance tracking."""
        try:
            logger = logging.getLogger(__name__)
            logger.info("Setting up CloudWatch logging")

            # Reduce noise from asyncio
            logging.getLogger("asyncio").setLevel(logging.WARNING)

            # Initialize CloudWatch client
            cloudwatch_client = boto3.client(
                "logs",
                aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
                region_name=config["AWS_REGION"],
            )

            # Determine a CloudWatch stream name. Prefer EC2 instance IDs when
            # available, but fall back to the hostname for Render and other
            # non-EC2 environments.
            stream_name = socket.gethostname()
            logger.debug("Retrieving EC2 instance ID")
            try:
                ec2_client = boto3.client(
                    "ec2",
                    region_name=config["AWS_REGION"],
                    aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
                    aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
                )
                describe_result = ec2_client.describe_instances()
                reservations = describe_result.get("Reservations", [])
                for reservation in reservations:
                    for instance in reservation.get("Instances", []):
                        instance_id = instance.get("InstanceId")
                        if instance_id:
                            stream_name = instance_id
                            break
                    if stream_name != socket.gethostname():
                        break
            except Exception as e:
                logger.warning(
                    f"Could not determine EC2 instance ID, using hostname fallback: {e}"
                )

            # Configure CloudWatch handler
            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group=config["CLOUDWATCH_LOG_GROUP"],
                stream_name=stream_name,
                boto3_client=cloudwatch_client,
                use_queues=False,
            )

            # Setup logging configuration
            logging.basicConfig(
                level=logging.DEBUG,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[cloudwatch_handler],
            )

            # Ensure logs are flushed on shutdown
            atexit.register(lambda: cloudwatch_handler.flush())

            logger.info(f"CloudWatch logging setup complete for stream {stream_name}")

        except Exception as e:
            logger.critical(f"Failed to setup CloudWatch logging: {e}")
            raise


class LLMManager:
    """OpenAI LLM initialization and management."""

    @staticmethod
    @retry(
        stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    def init_llm(config: Dict[str, Any], secret: Dict[str, Any]) -> OpenAI:
        """Initialize LLM model with retry logic."""
        try:
            model_type = config["LLM_MODEL_TYPE"]
            logger.info(f"Loading LLM model: {config[model_type]['MODEL_NAME']}")

            if model_type == "OPENAI":
                return OpenAI(
                    model=config["OPENAI"]["MODEL_NAME"],
                    api_key=secret["OPENAI_API_KEY"],
                    temperature=config["OPENAI"]["TEMPERATURE"],
                    top_p=config["OPENAI"]["TOP_P"],
                    max_tokens=config["OPENAI"]["MAX_TOKENS"],
                    timeout=config["OPENAI"]["REQUEST_TIMEOUT"],
                )
            elif model_type == "OLLAMA":
                return Ollama(
                    model=config["OLLAMA"]["MODEL_NAME"],
                    temperature=config["OLLAMA"]["TEMPERATURE"],
                    top_p=config["OLLAMA"]["TOP_P"],
                    request_timeout=config["OLLAMA"]["REQUEST_TIMEOUT"],
                    additional_kwargs={
                        "num_ctx": config["OLLAMA"]["CTX_LENGTH"],
                        "num_predict": config["OLLAMA"]["PREDICT_LENGTH"],
                        "cache": False,
                    },
                )
            elif model_type == "ANTHROPIC":
                return Anthropic(
                    model=config["ANTHROPIC"]["MODEL_NAME"],
                    api_key=secret["ANTHROPIC_API_KEY"],
                    temperature=config["ANTHROPIC"]["TEMPERATURE"],
                    max_tokens=config["ANTHROPIC"]["MAX_TOKENS"],
                    timeout=config["ANTHROPIC"]["REQUEST_TIMEOUT"],
                )
            elif model_type == "GROQ":
                return OpenAILike(
                    model=config["GROQ"]["MODEL_NAME"],
                    api_key=secret["GROQ_API_KEY"],
                
                    # Groq OpenAI-compatible endpoint
                    api_base=config["GROQ"]["API_BASE"],
                
                    temperature=config["GROQ"]["TEMPERATURE"],
                    max_tokens=config["GROQ"]["MAX_TOKENS"],
                    timeout=config["GROQ"]["REQUEST_TIMEOUT"],
                    additional_kwargs={"reasoning_effort": "low"},
                    is_chat_model=True,
                    reuse_client=True,
                )
            else:
                raise ValueError(f"Invalid LLM model type: {model_type}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise


class AppManager:
    _settings = None
    _llm = None
    _prompts = None
    _app = None
    _initialized = False
    _startup_task = None
    _startup_error = None
    _startup_event = None
    _qdrant_client = None
    _async_qdrant_client = None
    _reranker = None

    def __init__(self):
        self._settings = None
        self._llm = None
        self._prompts = None
        self._app = None
        self._initialized = False
        self._startup_task = None
        self._startup_error = None
        self._startup_event = asyncio.Event()
        self._qdrant_client = None
        self._async_qdrant_client = None
        self._reranker = None

    def is_ready(self) -> bool:
        return self._initialized

    def start_initialization(self) -> None:
        if self._initialized or self._startup_task is not None:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._initialize_async())
            return

        self._startup_task = loop.create_task(self._initialize_async())

    async def wait_until_ready(self) -> None:
        if self._initialized:
            return
        if self._startup_task is None:
            self.start_initialization()
        await self._startup_event.wait()
        if self._startup_error is not None:
            raise self._startup_error

    async def _initialize_async(self) -> None:
        try:
            logger.info("Initializing application services")
            self._settings = Settings()
            # Always retain a local, structured copy. CloudWatch remains an
            # additional destination when it is available.
            configure_local_logging()
            try:
                LogManager.setup_logging(self._settings.config, self._settings.secret)
            except Exception as exc:
                logger.warning(
                    "CloudWatch logging initialization failed; continuing without CloudWatch logging: %s",
                    exc,
                )
            self._prompts = await asyncio.to_thread(
                PromptManager.load_prompts, self._settings.config
            )
            self._llm = await asyncio.to_thread(
                LLMManager.init_llm, self._settings.config, self._settings.secret
            )
            from llama_index.core import Settings as LlamaSettings
            from llama_index.embeddings.fastembed import FastEmbedEmbedding
            from pathlib import Path

            repo_root = Path(__file__).resolve().parent
            abs_cache_dir = str(repo_root / ".fastembed_cache")
            
            logger.info(f"Globally initializing embedding model from: {abs_cache_dir}")
            global_embed = FastEmbedEmbedding(
                model_name=self._settings.config["HF_EMBED"],
                max_length=self._settings.config.get("HF_EMBED_MAX_LENGTH", 512),
                cache_dir=abs_cache_dir
            )

            # Lock them globally inside LlamaIndex settings
            LlamaSettings.llm = self._llm
            LlamaSettings.embed_model = global_embed
            # Reuse connection-owning clients for every request in this worker.
            # Keep the lightweight startup test configuration valid when these
            # optional external-service settings are intentionally absent.
            qdrant_url = self._settings.secret.get("QDRANT_URL")
            cohere_key = self._settings.secret.get("COHERE_API_KEY")
            if qdrant_url:
                self._qdrant_client = qdrant_client.QdrantClient(
                    url=qdrant_url,
                    api_key=self._settings.secret.get("QDRANT_API_KEY"),
                )
                self._async_qdrant_client = qdrant_client.AsyncQdrantClient(
                    url=qdrant_url,
                    api_key=self._settings.secret.get("QDRANT_API_KEY"),
                )
            else:
                logger.warning("Qdrant client not initialized: QDRANT_URL is missing")
            if cohere_key:
                self._reranker = ConditionalCohereRerank(
                    api_key=cohere_key,
                    model=self._settings.config["COHERE_RERANKER"],
                    top_n=self._settings.config["RAG_RERANKED_TOP_N"],
                )
            else:
                logger.warning("Cohere reranker not initialized: COHERE_API_KEY is missing")
            # --------------------------------------------------------
            self._initialized = True
        except Exception as exc:
            logger.exception("Application initialization failed: %s", exc)
            self._startup_error = exc
        finally:
            self._startup_event.set()

    def initialize(self) -> None:
        if self._initialized:
            return
        if self._startup_task is None:
            self.start_initialization()

    @property
    def settings(self):
        if self._settings is None:
            self.initialize()
        return self._settings

    @property
    def app(self):
        if AppManager._app is None:
            logger.info("Creating FastAPI application")

            AppManager._app = FastAPI(
                title="Aeronation API",
                version="1.0",
                description="API for Aeronation RAG system",
                lifespan=lifespan,

            )

            AppManager._app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        return AppManager._app

    @property
    def prompts(self):
        if self._prompts is None:
            self.initialize()
        return self._prompts

    @property
    def llm(self):
        if self._llm is None:
            self.initialize()
        return self._llm

    @property
    def qdrant_client(self):
        return self._qdrant_client

    @property
    def async_qdrant_client(self):
        return self._async_qdrant_client

    @property
    def reranker(self):
        return self._reranker


@asynccontextmanager
async def lifespan(app):
    try:
        yield
    finally:
        logger.info("Shutting down application")


app_manager = AppManager()
app = app_manager.app


async def save_chat_history(
    chat_id: str, chat_hist: str, config: Dict[str, Any]
) -> None:
    """Background task for chat history summarization and saving to S3."""
    try:
        await app_manager.wait_until_ready()
        start_time = time.perf_counter()
        logger.info(f"Starting chat history summarization for chat {chat_id}")

        chat_summ_file = f"{config['S3_CHAT_HISTORY']}/{chat_id}.md"

        # Generate chat summary
        logger.debug("Generating chat summary")
        summarized_hist = (await app_manager.llm.acomplete(
            app_manager.prompts.history_summarizer.format(chat_history=chat_hist), max_tokens=256
        )).text.strip()

        # Save to S3
        logger.debug("Saving summary to S3")
        s3_client = boto3.client("s3")
        await asyncio.to_thread(
            s3_client.put_object,
            Bucket=config["S3_LOGS_BUCKET"], Key=chat_summ_file, Body=summarized_hist,
        )

        duration = time.perf_counter() - start_time
        logger.info(
            f"Chat history summarization completed for chat {chat_id} in {duration:.2f} seconds"
        )

    except ClientError as e:
        logger.error(
            f"S3 error during chat history summarization for chat {chat_id}: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error during chat history summarization for chat {chat_id}: {e}"
        )
        raise


@app.get("/health", tags=["Health Check"])
async def health_check() -> JSONResponse:
    """Basic health check endpoint that answers instantly in <1ms."""
    return JSONResponse(status_code=200, content={"status": "OK"})




@app.post("/v1/chat", tags=["Chat API"])
async def get_answer(rag: RAG, background_tasks: BackgroundTasks) -> StreamingResponse:
    """Process chat requests and generate RAG-based responses."""
    try:
        if not app_manager.is_ready():
            logger.info("First query received. Lazily initializing all application models...")
            await app_manager.wait_until_ready()

        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        logger.info(f"Processing chat request for chat_id: {rag.chat_id}")
        logger.info(
            f"API request started in {time.perf_counter() - start_time:.2f} seconds"
        )

        # These Groq calls are independent, so run them on the shared async client
        # at the same time rather than serialising two network round trips.
        validation_started = time.perf_counter()
        profanity_call = app_manager.llm.acomplete(
            app_manager.prompts.profanity_filter.format(query=rag.query), max_tokens=32
        )
        rephrase_started = time.perf_counter()
        rephrase_call = app_manager.llm.acomplete(
            app_manager.prompts.rephrased_query.format(query=rag.query), max_tokens=64
        )
        profanity_result, rephrase_result = await asyncio.gather(profanity_call, rephrase_call)
        logger.info("stage=validation duration=%.3fs", time.perf_counter() - validation_started)
        log_latency("validation", time.perf_counter() - validation_started, request_id=request_id, chat_id=rag.chat_id)
        logger.info("stage=rephrase duration=%.3fs", time.perf_counter() - rephrase_started)
        log_latency("rephrase", time.perf_counter() - rephrase_started, request_id=request_id, chat_id=rag.chat_id)
        if profanity_result.text.strip() == "True":
            logger.warning("Inappropriate content detected in query for chat_id=%s", rag.chat_id)
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail="Sorry, I won't be able to answer your query.",
            )
        rag.query = rephrase_result.text.strip()
        logger.info(f"Updated Query: {rag.query}")

        # Update metadata
        metadata = {}
        if rag.file_name:
            metadata["file_name"] = rag.file_name

        # Initialize storage manager and response generator
        logger.debug("Initializing response generator")
        s3_manager = StorageManager(app_manager.settings.config, app_manager.settings.secret)

        generator_started = time.perf_counter()
        generate_obj = await Generate.create(
            config=app_manager.settings.config,
            secret=app_manager.settings.secret,
            chat_id=rag.chat_id,
            query=rag.query,
            persist_dir=rag.persist_dir,
            collection_name=rag.collection_name,
            s3_manager=s3_manager,
            metadata=metadata,
            qdrant_client_instance=app_manager.qdrant_client,
            async_qdrant_client_instance=app_manager.async_qdrant_client,
            reranker=app_manager.reranker,
            request_id=request_id,
        )
        logger.info("stage=generator_setup duration=%.3fs", time.perf_counter() - generator_started)
        log_latency("generator_setup", time.perf_counter() - generator_started, request_id=request_id, chat_id=rag.chat_id)

        # Schedule chat history summarization
        logger.debug("Scheduling chat history summarization")
        background_tasks.add_task(
            save_chat_history, rag.chat_id, s3_manager.chat_hist, app_manager.settings.config
        )

        # Generate streaming response
        logger.debug("Starting response generation")
        response = generate_obj.generate_answer()
        if response is not None:
            duration = time.perf_counter() - start_time
            logger.info(
                f"Chat request processed successfully in {duration:.2f} seconds"
            )
            return StreamingResponse(content=response, media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )
    finally:  
        logger.info("Executing post-query memory sweep...")
        
        # Explicitly delete query generation object handles to release context
        if 'generate_obj' in locals():
            del generate_obj
        if 's3_manager' in locals():
            del s3_manager
            
        # Forcibly collect untracked data loops and clear memory overhead
        gc.collect() 
        logger.info("Memory sweep complete.")


if __name__ == "__main__":
    import uvicorn
    import os 

    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    port = int(os.environ.get("PORT", 8000))

    logger.info("Starting uvicorn server")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,
        # Native Windows worker processes are unstable with this model stack.
        # Keep local development on one worker; Render's Linux command uses two.
        workers=1,
        loop="asyncio"
    )
