"""Service layer for the AeroBook AI routes.

No FastAPI concerns live here. This module adapts the existing RAG
implementation in generate.py to the endpoint-specific API in routes.py and
raises ordinary exceptions for the router to translate into status codes.

"""

import json
import logging
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config" / "config.yaml"
UPLOAD_DIR = PROJECT_DIR / "uploads"
PERSIST_PREFIX = "persist"

# Single-process only. Move to Redis or a database before running more than
# one uvicorn worker, otherwise a job created by worker A is invisible to B.
JOBS: Dict[str, Dict[str, Any]] = {}

_jobs_lock = threading.Lock()
_ingest_lock = threading.Lock()
_runtime_lock = threading.Lock()

_runtime_cache: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
_prompt_cache: Dict[str, str] = {}
_s3_cache = None


class ContentRejected(Exception):
    """The query was blocked by the profanity filter."""


# --------------------------------------------------------------------------- #
# Shared runtime
# --------------------------------------------------------------------------- #

def get_runtime() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Config and secrets, loaded once per process.

    The previous version called Secrets Manager on every request, which added
    an AWS round trip to the latency of every endpoint.
    """
    global _runtime_cache
    if _runtime_cache is None:
        with _runtime_lock:
            if _runtime_cache is None:
                from secrets_manager import get_secret

                with CONFIG_PATH.open(encoding="utf-8") as config_file:
                    config = yaml.safe_load(config_file)

                # generate.py resolves prompt paths from this setting, so make
                # it absolute and independent of uvicorn's working directory.
                config["PROMPT_DIR"] = str(PROJECT_DIR / config["PROMPT_DIR"])

                secret = get_secret(config)
                if not secret:
                    raise RuntimeError(
                        "Secrets Manager returned nothing. Check AWS credentials and "
                        "SECRETS_MANAGER in config/config.yaml."
                    )

                _runtime_cache = (config, secret)
                logger.info("Runtime config and secrets loaded")
    return _runtime_cache


def load_prompt(file_name: str) -> str:
    """Read a prompt template from PROMPT_DIR, cached after first read."""
    if file_name not in _prompt_cache:
        config, _ = get_runtime()
        path = Path(config["PROMPT_DIR"]) / file_name
        _prompt_cache[file_name] = path.read_text(encoding="utf-8").strip()
        logger.info("Loaded prompt %s", file_name)
    return _prompt_cache[file_name]


def get_llm():
    """ModelManager caches the client at class level, so this is cheap."""
    from generate import ModelManager

    config, secret = get_runtime()
    return ModelManager(config, secret).llm_model


def get_s3():
    global _s3_cache
    if _s3_cache is None:
        import boto3

        config, secret = get_runtime()
        _s3_cache = boto3.client(
            "s3",
            aws_access_key_id=secret["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"],
            region_name=config["AWS_REGION"],
        )
    return _s3_cache


# --------------------------------------------------------------------------- #
# Query pre-processing (ported from the deleted /v1/chat handler in app.py)
# --------------------------------------------------------------------------- #

def check_query(query: str) -> None:
    """Raise ContentRejected if the profanity filter flags the query."""
    try:
        verdict = get_llm().complete(
            load_prompt("profanity_filter.prompt").format(query=query), max_tokens=32
        ).text.strip()
    except Exception as exc:
        # A filter outage should not take the endpoint down. Log and allow.
        logger.error("Profanity filter failed, allowing query through: %s", exc, exc_info=True)
        return

    if verdict.lower().startswith("true"):
        logger.warning("Query rejected by profanity filter: %r", query[:120])
        raise ContentRejected("Sorry, I cannot answer that query.")


def rephrase(query: str) -> str:
    """Rewrite the query for retrieval. Falls back to the original on failure."""
    try:
        rewritten = get_llm().complete(
            load_prompt("rephrased_query.prompt").format(query=query), max_tokens=64
        ).text.strip()
        if rewritten:
            logger.info("Rephrased query: %r -> %r", query[:80], rewritten[:80])
            return rewritten
    except Exception as exc:
        logger.error("Query rephrasing failed, using original: %s", exc, exc_info=True)
    return query


def _build_generator(query: str, chat_id: str, collection_name: str):
    """Construct the existing RAG generator. Returns (generator, storage_manager)."""
    from generate import Generate, StorageManager

    config, secret = get_runtime()
    storage_manager = StorageManager(config, secret)
    generator = Generate(
        config=config,
        secret=secret,
        chat_id=chat_id,
        query=query,
        persist_dir=PERSIST_PREFIX,
        collection_name=collection_name,
        s3_manager=storage_manager,
    )
    return generator, storage_manager


def _save_chat_history(chat_id: str, storage_manager) -> None:
    """Summarise and persist chat history to S3. Never raises."""
    try:
        history = storage_manager.chat_hist
        if not history:
            return

        config, _ = get_runtime()
        summary = get_llm().complete(
            load_prompt("history_summarizer.prompt").format(chat_history=history),
            max_tokens=256,
        ).text.strip()

        get_s3().put_object(
            Bucket=config["S3_LOGS_BUCKET"],
            Key=f"{config['S3_CHAT_HISTORY']}/{chat_id}.md",
            Body=summary.encode("utf-8"),
        )
        logger.info("Saved chat history for %s", chat_id)
    except Exception as exc:
        logger.error("Could not save chat history for %s: %s", chat_id, exc, exc_info=True)


# --------------------------------------------------------------------------- #
# ask
# --------------------------------------------------------------------------- #

def answer_question(
    query: str, chat_id: str, collection_name: str
) -> Tuple[str, List[Dict[str, Any]]]:
    """Return the completed answer and its citation metadata."""
    check_query(query)
    generator, storage_manager = _build_generator(rephrase(query), chat_id, collection_name)

    answer = ""
    sources: List[Dict[str, Any]] = []

    for chunk in generator.generate_answer():
        event = json.loads(chunk)
        kind = event.get("type")
        if kind == "tokens":
            # The web-search fallback path emits only tokens, never an answer block.
            answer += event.get("text") or ""
        elif kind == "answer":
            answer = event.get("text", answer)
        elif kind == "context":
            sources = list(json.loads(event.get("text", "{}")).values())

    _save_chat_history(chat_id, storage_manager)
    return answer.strip(), sources


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #

def save_upload(file, job_id: str) -> Path:
    """Write the uploaded file to a job-specific path and return that path."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"{job_id}.pdf"
    with path.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    logger.info("Saved upload %s (%.2f MB)", path, path.stat().st_size / (1024 * 1024))
    return path


def create_job(job_id: str, file_name: str, chapter_id: str, collection_name: str) -> None:
    with _jobs_lock:
        JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "file_name": file_name,
            "chapter_id": chapter_id,
            "collection_name": collection_name,
            "queued_at": time.time(),
        }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def run_ingest(job_id: str, pdf_path: Path, chapter_id: str, collection_name: str) -> None:
    """Background worker. Must never raise: nothing is listening for it."""
    with _jobs_lock:
        if job_id not in JOBS:
            # Do not raise here. A background task exception goes nowhere useful.
            logger.error("Ingest requested for unknown job %s", job_id)
            pdf_path.unlink(missing_ok=True)
            return
        JOBS[job_id].update(status="processing", started_at=time.time())

    logger.info("Ingest job %s started: %s", job_id, pdf_path.name)
    try:
        result = index_pdf(pdf_path, chapter_id, collection_name)
        with _jobs_lock:
            JOBS[job_id].update(status="completed", result=result, finished_at=time.time())
        logger.info("Ingest job %s completed: %s", job_id, result)
    except Exception as exc:
        with _jobs_lock:
            JOBS[job_id].update(status="failed", error=str(exc), finished_at=time.time())
        logger.exception("Ingest job %s failed", job_id)
    finally:
        pdf_path.unlink(missing_ok=True)


def _split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size character chunks with overlap, no duplicated tail."""
    step = chunk_size - overlap
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunks.append(text[start:end])
        if end == length:
            break
        start += step
    return chunks


def index_pdf(pdf_path: Path, chapter_id: str, collection_name: str) -> Dict[str, Any]:
    """Extract a PDF, index its chunks in Qdrant, and upload persistence to S3."""
    from llama_index.core import Settings, StorageContext, load_index_from_storage
    from llama_index.core.indices.vector_store.base import VectorStoreIndex
    from llama_index.core.node_parser.node_utils import build_nodes_from_splits
    from llama_index.core.schema import Document
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.qdrant import QdrantVectorStore
    from generate import StorageManager
    import qdrant_client

    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader

    config, secret = get_runtime()

    reader = PdfReader(str(pdf_path))
    documents = []
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Page %s of %s failed to extract: %s", page_number, pdf_path.name, exc)
            continue
        if text.strip():
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": pdf_path.stem,
                        "page_num": page_number,
                        "chapter_id": chapter_id,
                    },
                )
            )

    if not documents:
        raise ValueError("The uploaded PDF contains no extractable text. It may be a scan needing OCR.")

    chunk_size = int(config["RAG_CITATION_CHUNK_SIZE"])
    overlap = int(config["RAG_CITATION_CHUNK_OVERLAP"])
    if not 0 <= overlap < chunk_size:
        raise ValueError("RAG_CITATION_CHUNK_OVERLAP must be non-negative and smaller than chunk size")

    nodes = []
    for document in documents:
        splits = _split(document.get_content(), chunk_size, overlap)
        if splits:
            nodes.extend(build_nodes_from_splits(splits, document))
    logger.info("Built %s nodes from %s pages of %s", len(nodes), len(documents), pdf_path.name)

    embedding_model = HuggingFaceEmbedding(model_name=config["HF_EMBED"])
    # load_index_from_storage resolves Settings.llm; without this LlamaIndex
    # falls back to its default OpenAI client and fails on a missing key.
    Settings.embed_model = embedding_model
    Settings.llm = get_llm()

    qdrant = qdrant_client.QdrantClient(url=secret["QDRANT_URL"], api_key=secret["QDRANT_API_KEY"])
    async_qdrant = qdrant_client.AsyncQdrantClient(url=secret["QDRANT_URL"], api_key=secret["QDRANT_API_KEY"])
    vector_store = QdrantVectorStore(
        collection_name=collection_name,
        client=qdrant,
        aclient=async_qdrant,
        enable_hybrid=config["QDRANT_ENABLE_HYBRID"],
        fastembed_sparse_model=config["FASTEMBED_SPARSE_MODEL"],
        prefer_grpc=False,
    )

    s3 = get_s3()
    bucket = config["S3_PERSIST_BUCKET"]
    prefix = f"{PERSIST_PREFIX}/{collection_name}"

    # The lock serialises ingests so two uploads cannot both read-modify-write
    # the same persist directory and clobber each other's docstore.
    with _ingest_lock, tempfile.TemporaryDirectory(prefix="aeronation-ingest-") as temp_dir:
        persist_dir = Path(temp_dir) / collection_name
        persist_dir.mkdir(parents=True, exist_ok=True)

        objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix).get("Contents", [])
        downloaded = 0
        for obj in objects:
            key = obj["Key"]
            relative = key.removeprefix(prefix).lstrip("/")
            if key.endswith("/") or not relative:
                continue  # directory marker, or the prefix key itself
            destination = persist_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(destination))
            downloaded += 1

        if downloaded:
            logger.info("Loaded existing index (%s files) for %s", downloaded, collection_name)
            storage_context = StorageContext.from_defaults(
                persist_dir=str(persist_dir), vector_store=vector_store
            )
            index = load_index_from_storage(storage_context)
        else:
            logger.info("No existing index for %s, creating a new one", collection_name)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes=[], storage_context=storage_context, embed_model=embedding_model
            )

        index.insert_nodes(nodes)
        storage_context.persist(persist_dir=str(persist_dir))

        uploaded = 0
        for local_file in persist_dir.rglob("*"):
            if local_file.is_file():
                key = f"{prefix}/{local_file.relative_to(persist_dir).as_posix()}"
                s3.upload_file(str(local_file), bucket, key)
                uploaded += 1

    # StorageManager caches downloaded persist directories at class level and
    # never expires them. Without this, /ask and /chat keep serving the index
    # from before this upload for the lifetime of the process.
    StorageManager._persist_dir_cache.pop(f"{PERSIST_PREFIX}/{collection_name}", None)
    logger.info("Invalidated persist cache for %s", collection_name)

    return {
        "file_name": pdf_path.name,
        "pages": len(documents),
        "chunks": len(nodes),
        "persist_files_uploaded": uploaded,
    }


# --------------------------------------------------------------------------- #
# generate and summarize
# --------------------------------------------------------------------------- #

def _complete(prompt: str, max_tokens: int) -> str:
    return get_llm().complete(prompt, max_tokens=max_tokens).text.strip()


def generate_questions(
    topic: str, count: int, difficulty: str, qtype: str, collection_name: str
) -> List[Dict[str, Any]]:
    """Generate structured questions grounded in retrieved corpus chunks.

    This retrieves directly rather than going through answer_question, so the
    questions are written from the source text instead of from a summary of it.
    """
    from llama_index.core.schema import QueryBundle

    generator, _ = _build_generator(topic, "question-generator", collection_name)
    retrieved = generator.query_engine.retrieve(QueryBundle(query_str=topic))
    logger.info("generate_questions: retrieved %s chunks for %r", len(retrieved), topic)
    if not retrieved:
        return []

    context = "\n\n".join(node.node.get_text() for node in retrieved)
    prompt = (
        "Create {count} {difficulty} {qtype} questions using only this context. "
        "Return a JSON array and nothing else. Each item must contain the keys "
        "question, options, correctAnswer and explanation. "
        "For non-MCQ types set options to an empty list.\n\n"
        "Context:\n{context}"
    ).format(count=count, difficulty=difficulty, qtype=qtype, context=context)

    response = _complete(prompt, max_tokens=max(512, count * 180))
    start, end = response.find("["), response.rfind("]")
    if start < 0 or end < start:
        logger.error("No JSON array in model response: %r", response[:500])
        raise json.JSONDecodeError("No JSON array in model response", response, 0)

    questions = json.loads(response[start:end + 1])
    if not isinstance(questions, list):
        raise json.JSONDecodeError("Model response is not a JSON array", response, 0)
    return questions


def summarize_highlights(highlights: List[str], style: str) -> str:
    """Summarize user-provided highlights in the requested presentation style."""
    instruction = "a concise bullet list" if style == "bullets" else "one concise paragraph"
    return _complete(
        "Summarize the following highlights as {instruction}. "
        "Keep every formula, number and technical term exactly as written. "
        "Do not add unsupported facts.\n\n{body}".format(
            instruction=instruction,
            body="\n".join("- " + highlight for highlight in highlights),
        ),
        max_tokens=512,
    )


# --------------------------------------------------------------------------- #
# chat
# --------------------------------------------------------------------------- #

def stream_chat(query: str, chat_id: str, collection_name: str) -> Iterator[str]:
    """Return an SSE event iterator for the RAG stream.

    This is deliberately a plain function that returns a generator, not a
    generator function itself. Setup (profanity check, index load, query engine
    construction) runs when the route calls this, so failures can still become a
    500. If this were a generator function nothing would execute until the first
    chunk was pulled, by which point the 200 response has already been sent.
    """
    check_query(query)
    generator, storage_manager = _build_generator(rephrase(query), chat_id, collection_name)
    stream = generator.generate_answer()

    def events() -> Iterator[str]:
        for event in stream:
            yield "data: " + event + "\n\n"
        _save_chat_history(chat_id, storage_manager)
        yield "data: [DONE]\n\n"

    return events()   