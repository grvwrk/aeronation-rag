# scripts/download_model.py
import os
import yaml
from pathlib import Path
from llama_index.embeddings.fastembed import FastEmbedEmbedding

# parents[1] moves out of 'scripts/' and targets the project root directory
repo_root = Path(__file__).resolve().parents[1]

cfg_path = repo_root / "config" / "config.yaml"
with open(cfg_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# This will now create the folder at the root project directory level
abs_cache_dir = str(repo_root / ".fastembed_cache")
print(f"Pre-downloading embedding model to absolute path: {abs_cache_dir}")

model = FastEmbedEmbedding(
    model_name=config.get("HF_EMBED"),
    cache_dir=abs_cache_dir
)
print("Embedding model successfully cached to local disk.")
