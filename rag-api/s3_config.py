# rag-api/s3_config.py
"""
S3 configuration loader from config/config.yaml, matching S3UploadConfig used by s3_upload.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

CONFIG_PATH = Path(__file__).parent / "config" / "config.yaml"


@dataclass
class S3UploadConfig:
    """
    Matches the S3UploadConfig expected by ingestion/shared_processing/s3_upload.py.
    """
    bucket_name: str
    base_prefix: str
    region_name: str
    staging_prefix: str = ".tmp"
    enable_staging: bool = True
    overwrite: bool = False
    collection_type: str | None = None


def load_config_yaml() -> Dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_s3_upload_config_from_yaml(
    cfg: Dict[str, Any] | None = None,
    collection_type: str | None = None,
) -> S3UploadConfig:
    """
    Build S3UploadConfig using values from config.yaml.

    Expected keys in YAML (top-level):
      - S3_PERSIST_BUCKET      -> bucket_name
      - S3_PERSIST_DIR         -> base_prefix (remote base path)
      - AWS_REGION             -> region_name
      - S3_ENABLE_STAGING      -> enable_staging (optional, default True)
    """
    if cfg is None:
        cfg = load_config_yaml()

    bucket_name = cfg.get("S3_PERSIST_BUCKET", "")
    base_prefix = cfg.get("S3_PERSIST_DIR", "persist")
    region_name = cfg.get("AWS_REGION", "ap-south-1")
    enable_staging = cfg.get("S3_ENABLE_STAGING", True)

    # Normalize base_prefix: strip leading/trailing slashes
    if base_prefix:
        base_prefix = base_prefix.strip("/")
    if not base_prefix:
        base_prefix = "persist"

    # Staging prefix under the collection, e.g. "<base_prefix>/<collection>/.tmp/..."
    staging_prefix = ".tmp"

    return S3UploadConfig(
        bucket_name=bucket_name,
        base_prefix=base_prefix,
        region_name=region_name,
        staging_prefix=staging_prefix,
        enable_staging=bool(enable_staging),
        overwrite=False,
        collection_type=collection_type,
    )