#!/usr/bin/env python3
"""
Create a LlamaIndex persistence directory for a Qdrant-backed collection from local PDFs
and upload the resulting `persist/<collection_name>/` files to S3 under the same prefix.

Usage: python scripts/create_persist.py --data-dir ./data --collection rag_llm

This script:
- loads PDFs from `data_dir`
- extracts text (requires `pypdf` / `PyPDF2`)
- chunks text into simple fixed-size chunks
- builds nodes and inserts them into a `QdrantVectorStore` via LlamaIndex
- persists the LlamaIndex doc/index files to `persist/<collection_name>/`
- uploads the persist files to S3 at `<S3_PERSIST_DIR>/<persist_dir>/<collection_name>/`

"""
import argparse
import logging
import math
import os
import sys
from pathlib import Path
import yaml

import boto3
import qdrant_client

# Ensure project root is on sys.path so sibling modules (e.g. secrets_manager.py)
# can be imported when running `python scripts/create_persist.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Compatibility shim for newer qdrant-client versions with llama-index-vector-stores-qdrant
try:
    import qdrant_client.qdrant_fastembed as _qf
    if not hasattr(_qf, "IDF_EMBEDDING_MODELS"):
        _qf.IDF_EMBEDDING_MODELS = []
except Exception:
    pass

from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core.readers.base import BasePydanticReader
from llama_index.core.schema import Document

try:
    # modern package name
    from pypdf import PdfReader
except Exception:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        PdfReader = None

logger = logging.getLogger("create_persist")
logging.basicConfig(level=logging.INFO)


class LocalPDFReader(BasePydanticReader):
    """Minimal PDF reader returning LlamaIndex `Document` objects."""

    def lazy_load_data(self, input_file: Path, extra_info=None, **kwargs):
        if PdfReader is None:
            raise ImportError(
                "No PDF reader available. Install `pypdf` or `PyPDF2` in your environment."
            )

        reader = PdfReader(str(input_file))
        docs = []
        num_pages = len(reader.pages)
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""

            meta = {} if extra_info is None else dict(extra_info)
            meta.update({"file_name": input_file.stem, "page_num": i + 1})
            docs.append(Document(text=text, metadata=meta))

        return docs


def chunk_text(text: str, chunk_size: int = 128, overlap: int = 8):
    if not text:
        return []
    if chunk_size <= 0:
        return [text]
    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start += step
    return chunks


def ensure_qdrant_collection(qclient: qdrant_client.QdrantClient, collection_name: str, vector_size: int):
    try:
        col_info = qclient.get_collection(collection_name)
        logger.info(f"Qdrant collection '{collection_name}' already exists")
    except Exception:
        logger.info(f"Creating Qdrant collection '{collection_name}' with dim={vector_size}")
        qclient.recreate_collection(collection_name=collection_name, vector_size=vector_size)


def upload_dir_to_s3(s3_client, bucket: str, local_dir: Path, s3_prefix: str):
    local_dir = Path(local_dir)
    for p in local_dir.rglob("*"):
        if p.is_file():
            rel = p.relative_to(local_dir)
            s3_key = f"{s3_prefix.rstrip('/')}/{rel.as_posix()}"
            logger.info(f"Uploading {p} -> s3://{bucket}/{s3_key}")
            s3_client.upload_file(str(p), bucket, s3_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Directory with PDF files")
    parser.add_argument("--collection", default="rag_llm", help="Qdrant collection name")
    parser.add_argument("--persist-prefix", default="persist", help="S3 persist prefix (default: persist)")
    parser.add_argument("--qdrant-url", dest="qdrant_url", default=None, help="Qdrant URL (overrides secrets/env)")
    parser.add_argument("--qdrant-api-key", dest="qdrant_api_key", default=None, help="Qdrant API key (overrides secrets/env)")
    parser.add_argument("--chunk-size", type=int, default=None, help="Chunk size in characters; defaults to config RAG_CITATION_CHUNK_SIZE or 128")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Chunk overlap in characters; defaults to config RAG_CITATION_CHUNK_OVERLAP or 8")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = repo_root / "config" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    default_chunk_size = int(config.get("RAG_CITATION_CHUNK_SIZE", 128))
    default_chunk_overlap = int(config.get("RAG_CITATION_CHUNK_OVERLAP", 8))
    chunk_size = args.chunk_size if args.chunk_size is not None else default_chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else default_chunk_overlap

    if chunk_size <= 0:
        raise ValueError("chunk-size must be a positive integer")
    if chunk_overlap < 0:
        raise ValueError("chunk-overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk-overlap must be less than chunk-size")

    logger.info(f"Using chunk size={chunk_size}, chunk overlap={chunk_overlap}")

    secret = None
    try:
        # try to use the project's secret manager if available
        from secrets_manager import get_secret

        secret = get_secret(config)
        if not secret:
            logger.warning("secrets_manager.get_secret returned no data; attempting direct boto3 Secrets Manager fetch")
            # attempt direct boto3 fetch as a fallback
            try:
                sm = boto3.client("secretsmanager", region_name=config.get("AWS_REGION"))
                resp = sm.get_secret_value(SecretId=config.get("SECRETS_MANAGER"))
                sec_str = resp.get("SecretString") or resp.get("SecretBinary")
                if sec_str:
                    import json

                    secret = json.loads(sec_str)
                    logger.info("Loaded secrets from Secrets Manager via boto3 fallback")
            except Exception as e:
                logger.warning(f"Direct boto3 Secrets Manager fetch failed: {e}")
    except Exception as e:
        logger.warning(f"Could not load secrets via secrets_manager.get_secret; falling back to env/aws defaults: {e}")

    # build embedding model
    embed_model_name = config.get("HF_EMBED")
    if not embed_model_name:
        raise ValueError("HF_EMBED not set in config/config.yaml")

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Prepare Qdrant client
    # Qdrant connection info: CLI args -> secrets -> env
    qdrant_url = args.qdrant_url or (secret.get("QDRANT_URL") if secret else None) or os.environ.get("QDRANT_URL")
    qdrant_api_key = args.qdrant_api_key or (secret.get("QDRANT_API_KEY") if secret else None) or os.environ.get("QDRANT_API_KEY")

    if not qdrant_url:
        raise ValueError(
            "QDRANT_URL not found. Provide it via --qdrant-url, set QDRANT_URL in environment, or store it in Secrets Manager."
        )

    qclient = qdrant_client.QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    aclient = qdrant_client.AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Create vector store wrapper
    vector_store = QdrantVectorStore(
        collection_name=args.collection,
        client=qclient,
        aclient=aclient,
        enable_hybrid=config.get("QDRANT_ENABLE_HYBRID", False),
        fastembed_sparse_model=config.get("FASTEMBED_SPARSE_MODEL"),
        prefer_grpc=False,
    )

    # create an in-memory storage context with the qdrant vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # read files
    data_dir = Path(args.data_dir)
    # Resolve common locations: provided path, project-root relative, parent-of-project (sibling) relative
    candidates = [data_dir, PROJECT_ROOT / data_dir, PROJECT_ROOT.parent / data_dir]
    resolved = None
    for c in candidates:
        if c.exists():
            resolved = c
            break

    if resolved is None:
        raise ValueError(f"Data directory not found: {data_dir} (checked: {', '.join(str(x) for x in candidates)})")

    data_dir = resolved

    file_extractor = {".pdf": LocalPDFReader()}
    reader = SimpleDirectoryReader(input_dir=str(data_dir), file_extractor=file_extractor, recursive=True)
    documents = reader.load_data()
    logger.info(f"Loaded {len(documents)} document pages from {data_dir}")

    # build nodes
    nodes = []
    for doc in documents:
        text = doc.get_content() if hasattr(doc, "get_content") else getattr(doc, "text", "")
        splits = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
        if not splits:
            continue
        built = build_nodes_from_splits(splits, doc)
        nodes.extend(built)

    logger.info(f"Built {len(nodes)} nodes (chunks)")

    # create index and insert nodes
    index = VectorStoreIndex(nodes=[], storage_context=storage_context, embed_model=embed_model)
    index.insert_nodes(nodes)

    # persist local files
    local_persist_dir = Path("persist") / args.collection
    local_persist_dir.mkdir(parents=True, exist_ok=True)
    storage_context.persist(persist_dir=str(local_persist_dir))
    logger.info(f"Persisted storage to {local_persist_dir}")

    # upload to S3 under configured bucket
    s3_bucket = config.get("S3_PERSIST_BUCKET")
    if not s3_bucket:
        raise ValueError("S3_PERSIST_BUCKET not set in config/config.yaml")

    aws_access_key = secret.get("AWS_ACCESS_KEY_ID") if secret else None
    aws_secret = secret.get("AWS_SECRET_ACCESS_KEY") if secret else None

    s3_client_kwargs = {}
    if aws_access_key and aws_secret:
        s3_client_kwargs.update(
            dict(aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret, region_name=config.get("AWS_REGION"))
        )

    s3 = boto3.client("s3", **s3_client_kwargs) if s3_client_kwargs else boto3.client("s3")

    s3_prefix = f"{args.persist_prefix.rstrip('/')}/{args.collection}"
    upload_dir_to_s3(s3, s3_bucket, local_persist_dir, s3_prefix)

    logger.info("Upload complete. Persist directory is available in S3.")


if __name__ == "__main__":
    main()
