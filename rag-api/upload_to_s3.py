# rag-api/upload_to_s3.py
"""
Upload persisted RAG index (persist/<collection>) to S3 using config from config/config.yaml.

Usage (from rag-api directory):
  python upload_to_s3.py --collection rag_llm
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from s3_config import build_s3_upload_config_from_yaml, S3UploadConfig

from ingestion.shared_processing.s3_upload import upload_persistence_to_s3  # type: ignore

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Upload persisted RAG index to S3 using config/config.yaml"
    )
    parser.add_argument(
        "--collection",
        default="rag_llm",
        help="Collection name (subdirectory under persist/), e.g. rag_llm",
    )
    parser.add_argument(
        "--persist-dir",
        default="persist",
        help="Local base persist directory (relative to rag-api root)",
    )
    args = parser.parse_args()

    collection_type = args.collection

    # Build S3 config from YAML
    s3_cfg: S3UploadConfig = build_s3_upload_config_from_yaml(
        collection_type=collection_type,
    )

    if not s3_cfg.bucket_name or not s3_cfg.bucket_name.strip():
        logger.error("S3_PERSIST_BUCKET is not set in config/config.yaml; aborting.")
        return

    # Local path to the collection artifacts
    persist_dir = Path(args.persist_dir) / collection_type
    if not persist_dir.exists():
        logger.error(f"Persist directory {persist_dir} not found. Run ingestion first.")
        return

    logger.info(
        f"Uploading {persist_dir} to s3://{s3_cfg.bucket_name}/{s3_cfg.base_prefix}/{collection_type}"
    )
    logger.info(
        f"S3 config: bucket={s3_cfg.bucket_name}, base_prefix={s3_cfg.base_prefix}, "
        f"region={s3_cfg.region_name}, enable_staging={s3_cfg.enable_staging}, "
        f"collection_type={s3_cfg.collection_type}"
    )

    result = upload_persistence_to_s3(
        persist_dir=persist_dir,
        config=s3_cfg,
    )

    if result.success:
        logger.info(
            "S3 upload succeeded: %d files, %d bytes. Uploaded keys: %s",
            result.files_uploaded,
            result.bytes_uploaded,
            result.uploaded_keys,
        )
    else:
        logger.error("S3 upload failed: %s", result.error)


if __name__ == "__main__":
    main()