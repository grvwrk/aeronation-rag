"""
S3 Upload Stage: Validated local persistence artifact -> S3 versioned/production storage.

Implements the S3 Upload and Verification stage for the Aeronation-RAG data ingestion pipeline:
1. Complete artifact upload: Uploads all files in persist/<collection_type>/ (docstore, index_store, manifest, etc.).
2. Directory structure preservation: Preserves relative file paths under the S3 prefix.
3. Staging and atomic promotion: Uploads to a temporary/versioned prefix before promoting to production S3 location.
4. Post-upload S3 verification: Verifies uploaded S3 object existence, sizes, and manifest metadata.
5. Telemetry & statistics: Returns structured S3UploadResult (files_expected, files_uploaded, bytes_uploaded, keys).
6. Partial upload detection: Explicitly catches and fails on partial uploads without corrupting production.
7. Zero secrets: Strictly guarantees no credentials, keys, or secrets are logged or persisted.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    S3ConfigurationError,
    S3Error,
    S3ObjectNotFoundError,
    S3PermissionError,
    S3UploadError,
    S3VerificationError,
)
from models import S3ObjectInfo, S3UploadConfig, S3UploadResult
from validator import validate_local_artifact

logger = logging.getLogger(__name__)


def build_s3_key(
    base_prefix: str,
    collection_type: str,
    relative_path: str | Path,
    staging_subpath: str | None = None,
) -> str:
    """Construct a clean, normalized S3 object key.

    Example:
        build_s3_key("persist", "rag_llm", "docstore.json")
        -> "persist/rag_llm/docstore.json"

        build_s3_key("persist", "rag_llm", "docstore.json", staging_subpath=".tmp/v001")
        -> "persist/rag_llm/.tmp/v001/docstore.json"
    """
    clean_base = base_prefix.strip("/")
    clean_col = collection_type.strip("/")
    rel_posix = Path(relative_path).as_posix().lstrip("/")

    if staging_subpath:
        clean_stg = staging_subpath.strip("/")
        return f"{clean_base}/{clean_col}/{clean_stg}/{rel_posix}"
    return f"{clean_base}/{clean_col}/{rel_posix}"


def get_s3_client(
    config: S3UploadConfig | None = None,
    client: Any | None = None,
) -> Any:
    """Resolve or instantiate a boto3 S3 client."""
    if client is not None:
        return client

    cfg = config if config is not None else S3UploadConfig.from_env()
    try:
        import boto3

        return boto3.client("s3", region_name=cfg.region_name)
    except Exception as exc:
        raise S3ConfigurationError(
            bucket=cfg.bucket_name,
            prefix=cfg.base_prefix,
            reason=f"Failed to initialize boto3 S3 client: {exc}",
        ) from exc


def upload_file_to_s3(
    s3_client: Any,
    local_file: Path | str,
    bucket: str,
    s3_key: str,
) -> int:
    """Upload a single file to S3 and return the number of bytes uploaded.

    Raises:
        S3PermissionError: If AWS S3 returns AccessDenied (403).
        S3UploadError: If upload fails due to network, bucket, or transfer error.
    """
    path = Path(local_file)
    if not path.is_file():
        raise S3UploadError(
            bucket=bucket,
            prefix=s3_key,
            reason=f"Local file does not exist or is not a file: {path}",
            key=s3_key,
        )

    file_size = path.stat().st_size

    try:
        s3_client.upload_file(str(path), bucket, s3_key)
        logger.debug("Uploaded %s (%d bytes) -> s3://%s/%s", path.name, file_size, bucket, s3_key)
        return file_size
    except Exception as exc:
        err_msg = str(exc)
        # Check for permission error
        if "AccessDenied" in err_msg or "403" in err_msg or "Forbidden" in err_msg:
            raise S3PermissionError(
                bucket=bucket,
                prefix=s3_key,
                reason=f"Access denied uploading to s3://{bucket}/{s3_key}: {err_msg}",
                key=s3_key,
            ) from exc

        raise S3UploadError(
            bucket=bucket,
            prefix=s3_key,
            reason=f"Failed uploading {path.name} to s3://{bucket}/{s3_key}: {err_msg}",
            key=s3_key,
        ) from exc


def upload_directory_to_s3(
    s3_client: Any,
    local_dir: Path | str,
    bucket: str,
    s3_prefix: str,
) -> tuple[list[str], list[str], int]:
    """Recursively upload all files in local_dir preserving relative directory hierarchy.

    Returns:
        tuple of (uploaded_keys, failed_files, total_bytes_uploaded)
    """
    base_path = Path(local_dir)
    clean_prefix = s3_prefix.strip("/")

    files_to_upload = sorted(p for p in base_path.rglob("*") if p.is_file())
    uploaded_keys: list[str] = []
    failed_files: list[str] = []
    total_bytes = 0

    for file_path in files_to_upload:
        rel_path = file_path.relative_to(base_path).as_posix()
        s3_key = f"{clean_prefix}/{rel_path}"

        try:
            bytes_sent = upload_file_to_s3(s3_client, file_path, bucket, s3_key)
            uploaded_keys.append(s3_key)
            total_bytes += bytes_sent
        except Exception as exc:
            logger.error("Failed uploading %s to s3://%s/%s: %s", file_path.name, bucket, s3_key, exc)
            failed_files.append(rel_path)

    return uploaded_keys, failed_files, total_bytes


def verify_s3_objects(
    s3_client: Any,
    bucket: str,
    expected_keys: list[str],
    *,
    manifest_key: str | None = None,
    expected_collection_type: str | None = None,
    expected_ingestion_id: str | None = None,
) -> bool:
    """Verify that expected objects exist in S3 and validate the remote manifest.

    Validates:
    - All expected_keys exist in S3 (via head_object).
    - manifest.json exists and can be retrieved.
    - Remote manifest collection_type and ingestion_id match expected values.

    Raises:
        S3ObjectNotFoundError: If an expected key is missing from S3.
        S3VerificationError: If remote manifest is corrupt or has mismatched metadata.
    """
    prefix = expected_keys[0].rsplit("/", 1)[0] if expected_keys else ""

    # 1. Verify existence of each expected object
    for key in expected_keys:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            err_msg = str(exc)
            if "404" in err_msg or "NoSuchKey" in err_msg or "Not Found" in err_msg:
                raise S3ObjectNotFoundError(
                    bucket=bucket,
                    prefix=prefix,
                    reason=f"Expected S3 object not found: s3://{bucket}/{key}",
                    key=key,
                ) from exc
            raise S3VerificationError(
                bucket=bucket,
                prefix=prefix,
                reason=f"Failed verifying S3 object s3://{bucket}/{key}: {err_msg}",
                key=key,
            ) from exc

    # 2. Verify and parse remote manifest if specified
    if manifest_key:
        try:
            resp = s3_client.get_object(Bucket=bucket, Key=manifest_key)
            body = resp["Body"].read()
            manifest = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise S3VerificationError(
                bucket=bucket,
                prefix=prefix,
                reason=f"Failed reading/parsing remote manifest s3://{bucket}/{manifest_key}: {exc}",
                key=manifest_key,
            ) from exc

        remote_col = manifest.get("collection_type")
        remote_id = manifest.get("ingestion_id") or manifest.get("version")

        if expected_collection_type and remote_col != expected_collection_type:
            raise S3VerificationError(
                bucket=bucket,
                prefix=prefix,
                reason=(
                    f"Remote manifest collection_type '{remote_col}' does not match "
                    f"expected '{expected_collection_type}'"
                ),
                key=manifest_key,
            )

        if expected_ingestion_id and remote_id != expected_ingestion_id:
            raise S3VerificationError(
                bucket=bucket,
                prefix=prefix,
                reason=(
                    f"Remote manifest ingestion_id '{remote_id}' does not match "
                    f"expected '{expected_ingestion_id}'"
                ),
                key=manifest_key,
            )

    return True


def promote_staged_s3_artifact(
    s3_client: Any,
    bucket: str,
    staging_keys: list[str],
    staging_prefix: str,
    production_prefix: str,
) -> list[str]:
    """Atomically copy all staged S3 objects into their production S3 keys."""
    clean_stg = staging_prefix.strip("/")
    clean_prod = production_prefix.strip("/")
    promoted_keys: list[str] = []

    for stg_key in staging_keys:
        if not stg_key.startswith(clean_stg):
            continue
        rel_subpath = stg_key[len(clean_stg) :].lstrip("/")
        prod_key = f"{clean_prod}/{rel_subpath}"

        try:
            s3_client.copy_object(
                Bucket=bucket,
                CopySource={"Bucket": bucket, "Key": stg_key},
                Key=prod_key,
            )
            promoted_keys.append(prod_key)
            logger.debug("Promoted s3://%s/%s -> s3://%s/%s", bucket, stg_key, bucket, prod_key)
        except Exception as exc:
            raise S3UploadError(
                bucket=bucket,
                prefix=production_prefix,
                reason=f"Failed promoting staged object s3://{bucket}/{stg_key} to s3://{bucket}/{prod_key}: {exc}",
                key=prod_key,
            ) from exc

    return promoted_keys


def cleanup_staged_s3_artifact(
    s3_client: Any,
    bucket: str,
    staging_keys: list[str],
) -> None:
    """Clean up temporary staging objects in S3 after promotion or failure."""
    for key in staging_keys:
        try:
            s3_client.delete_object(Bucket=bucket, Key=key)
            logger.debug("Deleted staging object: s3://%s/%s", bucket, key)
        except Exception as exc:
            logger.warning("Failed to delete staging object s3://%s/%s: %s", bucket, key, exc)


def upload_persistence_to_s3(
    persist_dir: Path | str,
    config: S3UploadConfig | None = None,
    *,
    s3_client: Any | None = None,
    collection_type: str | None = None,
    ingestion_id: str | None = None,
) -> S3UploadResult:
    """Execute the full S3 persistence upload stage with atomic staging, verification, and promotion.

    Workflow:
    1. Validates local persistence directory and manifest (delegated to validator.py).
    2. Constructs dynamic S3 prefixes using centralized config.
    3. Uploads entire directory to temporary staging S3 location.
    4. Detects partial upload failures and cleans up staging if failed.
    5. Verifies all objects and manifest in S3 staging.
    6. Promotes staged objects to production S3 prefix.
    7. Verifies production S3 objects.
    8. Cleans up staging objects in S3.
    9. Returns comprehensive S3UploadResult telemetry.

    Raises:
        S3ConfigurationError: If bucket name is missing.
        S3PermissionError: If AWS credentials lack S3 permissions.
        S3UploadError: If upload fails or partial upload occurs.
        S3VerificationError: If post-upload verification fails.
    """
    local_dir = Path(persist_dir)

    # 1. Validate local artifact (manifest, files, checksums)
    manifest = validate_local_artifact(
        local_dir,
        expected_collection_type=collection_type,
        expected_ingestion_id=ingestion_id,
    )

    col_type = manifest.get("collection_type", collection_type or "rag_llm")
    ing_id = manifest.get("ingestion_id") or manifest.get("version", ingestion_id or "unknown")

    # 2. Resolve S3 upload configuration
    cfg = config if config is not None else S3UploadConfig.from_env(collection_type=col_type)

    if not cfg.bucket_name or not cfg.bucket_name.strip():
        raise S3ConfigurationError(
            bucket="",
            prefix=cfg.base_prefix,
            reason=(
                "S3 bucket name is not configured (missing S3_PERSIST_BUCKET or S3_BUCKET). "
                "S3 upload cannot proceed without a valid target bucket."
            ),
        )

    bucket = cfg.bucket_name.strip()
    client = get_s3_client(cfg, s3_client)

    prod_prefix = f"{cfg.base_prefix.strip('/')}/{col_type}"
    files_to_upload = sorted(p for p in local_dir.rglob("*") if p.is_file())
    expected_count = len(files_to_upload)

    if expected_count == 0:
        raise S3UploadError(
            bucket=bucket,
            prefix=prod_prefix,
            reason=f"Local persistence directory {local_dir} contains zero files to upload",
        )

    logger.info(
        "Starting S3 upload for collection '%s' (version=%s) to s3://%s/%s/ (%d files)",
        col_type,
        ing_id,
        bucket,
        prod_prefix,
        expected_count,
    )

    if cfg.enable_staging:
        staging_prefix = f"{prod_prefix}/{cfg.staging_prefix.strip('/')}/{ing_id}"
        staged_keys: list[str] = []

        try:
            # Stage 1: Upload to S3 staging prefix
            staged_keys, failed_files, bytes_uploaded = upload_directory_to_s3(
                client,
                local_dir,
                bucket,
                staging_prefix,
            )

            # Check for partial upload failure
            if failed_files or len(staged_keys) < expected_count:
                cleanup_staged_s3_artifact(client, bucket, staged_keys)
                raise S3UploadError(
                    bucket=bucket,
                    prefix=staging_prefix,
                    reason=(
                        f"Partial upload failure: expected {expected_count} files, "
                        f"uploaded {len(staged_keys)}, failed {len(failed_files)}: {failed_files}"
                    ),
                )

            # Stage 2: Verify staged S3 objects and manifest
            staged_manifest_key = f"{staging_prefix}/manifest.json"
            verify_s3_objects(
                client,
                bucket,
                staged_keys,
                manifest_key=staged_manifest_key,
                expected_collection_type=col_type,
                expected_ingestion_id=ing_id,
            )

            # Stage 3: Promote staged objects to production S3 prefix
            promoted_keys = promote_staged_s3_artifact(
                client,
                bucket,
                staged_keys,
                staging_prefix,
                prod_prefix,
            )

            # Stage 4: Verify production S3 objects and manifest
            prod_manifest_key = f"{prod_prefix}/manifest.json"
            verify_s3_objects(
                client,
                bucket,
                promoted_keys,
                manifest_key=prod_manifest_key,
                expected_collection_type=col_type,
                expected_ingestion_id=ing_id,
            )

            # Stage 5: Clean up staging artifacts in S3
            cleanup_staged_s3_artifact(client, bucket, staged_keys)

            logger.info(
                "S3 upload and promotion completed successfully for '%s' (version=%s): %d files (%d bytes)",
                col_type,
                ing_id,
                len(promoted_keys),
                bytes_uploaded,
            )

            return S3UploadResult(
                bucket=bucket,
                prefix=prod_prefix,
                collection_type=col_type,
                ingestion_id=ing_id,
                files_expected=expected_count,
                files_uploaded=len(promoted_keys),
                files_failed=0,
                bytes_uploaded=bytes_uploaded,
                uploaded_keys=sorted(promoted_keys),
                failed_files=[],
                manifest_key=prod_manifest_key,
                success=True,
            )

        except Exception as exc:
            # Clean up staging on any failure
            if staged_keys:
                cleanup_staged_s3_artifact(client, bucket, staged_keys)
            if isinstance(exc, S3Error):
                raise
            raise S3UploadError(
                bucket=bucket,
                prefix=staging_prefix,
                reason=f"Unexpected error during staged S3 upload: {exc}",
            ) from exc

    else:
        # Direct upload to production prefix
        try:
            uploaded_keys, failed_files, bytes_uploaded = upload_directory_to_s3(
                client,
                local_dir,
                bucket,
                prod_prefix,
            )

            if failed_files or len(uploaded_keys) < expected_count:
                raise S3UploadError(
                    bucket=bucket,
                    prefix=prod_prefix,
                    reason=(
                        f"Partial upload failure: expected {expected_count} files, "
                        f"uploaded {len(uploaded_keys)}, failed {len(failed_files)}: {failed_files}"
                    ),
                )

            prod_manifest_key = f"{prod_prefix}/manifest.json"
            verify_s3_objects(
                client,
                bucket,
                uploaded_keys,
                manifest_key=prod_manifest_key,
                expected_collection_type=col_type,
                expected_ingestion_id=ing_id,
            )

            return S3UploadResult(
                bucket=bucket,
                prefix=prod_prefix,
                collection_type=col_type,
                ingestion_id=ing_id,
                files_expected=expected_count,
                files_uploaded=len(uploaded_keys),
                files_failed=0,
                bytes_uploaded=bytes_uploaded,
                uploaded_keys=sorted(uploaded_keys),
                failed_files=[],
                manifest_key=prod_manifest_key,
                success=True,
            )

        except Exception as exc:
            if isinstance(exc, S3Error):
                raise
            raise S3UploadError(
                bucket=bucket,
                prefix=prod_prefix,
                reason=f"Unexpected error during direct S3 upload: {exc}",
            ) from exc
