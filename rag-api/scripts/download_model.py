# scripts/download_model.py
import os
import yaml
from pathlib import Path
from llama_index.embeddings.fastembed import FastEmbedEmbedding

repo_root = Path(__file__).resolve().parents[1]

cfg_path = repo_root / "config" / "config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Force an unmistakable absolute path structure
abs_cache_dir = str(repo_root / ".fastembed_cache")
print(f"Pre-downloading embedding model to absolute path: {abs_cache_dir}")

model = FastEmbedEmbedding(
    model_name=config.get("HF_EMBED"),
    cache_dir=abs_cache_dir  # <-- Absolute path
)
print("Embedding model successfully cached to local disk.")
