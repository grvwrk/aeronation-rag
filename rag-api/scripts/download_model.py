# scripts/download_model.py
import os
import yaml
from pathlib import Path
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from fastembed import SparseTextEmbedding  # Required to cache sparse models

# parents[1] moves out of 'scripts/' into the root project directory
repo_root = Path(__file__).resolve().parents[1]

cfg_path = repo_root / "config" / "config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Establish the exact same absolute path structure used by the live runtime app
abs_cache_dir = str(repo_root / ".fastembed_cache")
print(f"Pre-downloading models to absolute cache path: {abs_cache_dir}")

# 1. Download and cache the Dense Embedding Model (all-MiniLM-L6-v2)
dense_model_name = config.get("HF_EMBED")
print(f"Caching Dense Model: {dense_model_name}...")
dense_model = FastEmbedEmbedding(
    model_name=dense_model_name,
    cache_dir=abs_cache_dir
)

# 2. Download and cache the Sparse Embedding Model (Qdrant/bm42-all-minilm-l6-v2-attentions)
sparse_model_name = config.get("FASTEMBED_SPARSE_MODEL")
if sparse_model_name:
    print(f"Caching Sparse Model: {sparse_model_name}...")
    sparse_model = SparseTextEmbedding(
        model_name=sparse_model_name,
        cache_dir=abs_cache_dir
    )

print("🎉 All required embedding models have been successfully cached to local disk.")
