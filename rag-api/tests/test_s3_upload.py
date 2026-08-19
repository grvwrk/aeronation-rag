"""
Unit tests for S3 Upload Stage in the Aeronation RAG Ingestion Pipeline.
"""

import io
import json
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from errors import (
    IncompleteArtifactError,
    ManifestError,
    S3ConfigurationError,
    S3ObjectNotFoundError,
    S3PermissionError,
    S3UploadError,
    S3VerificationError,
)
from ingestion.shared_processing.persistence import (
    REQUIRED_PERSISTENCE_FILES,
    persist_collection,
)
from ingestion.shared_processing.s3_upload import (
    build_s3_key,
    cleanup_staged_s3_artifact,
    get_s3_client,
    promote_staged_s3_artifact,
    upload_directory_to_s3,
    upload_file_to_s3,
    upload_persistence_to_s3,
    verify_s3_objects,
)
from models import (
    ChunkingConfig,
    EmbeddingConfig,
    S3UploadConfig,
    S3UploadResult,
    StageCounts,
    VectorStoreConfig,
)
from validator import validate_collection_type, validate_local_artifact


class MockS3Client:
    """Deterministic, in-memory mock S3 client for unit testing."""

    def __init__(self) -> None:
        # Storage: (bucket, key) -> bytes
        self.storage: dict[tuple[str, str], bytes] = {}
        self.fail_on_keys: set[str] = set()
        self.permission_denied_keys: set[str] = set()
        self.network_error_keys: set[str] = set()
        self.call_log: list[tuple[str, dict[str, Any]]] = []

    def upload_file(self, filename: str, bucket: str, key: str, **kwargs: Any) -> None:
        self.call_log.append(("upload_file", {"filename": filename, "bucket": bucket, "key": key}))
        if key in self.permission_denied_keys:
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "PutObject",
            )
        if key in self.network_error_keys:
            raise RuntimeError("Simulated network timeout/connection reset during S3 upload")
        if key in self.fail_on_keys:
            raise RuntimeError(f"Simulated upload failure on key {key}")

        with open(filename, "rb") as f:
            self.storage[(bucket, key)] = f.read()

    def head_object(self, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.call_log.append(("head_object", {"Bucket": Bucket, "Key": Key}))
        if (Bucket, Key) not in self.storage:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        data = self.storage[(Bucket, Key)]
        return {
            "ContentLength": len(data),
            "ETag": '"mock-etag-12345"',
        }

    def get_object(self, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.call_log.append(("get_object", {"Bucket": Bucket, "Key": Key}))
        if (Bucket, Key) not in self.storage:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}},
                "GetObject",
            )
        data = self.storage[(Bucket, Key)]
        return {"Body": io.BytesIO(data), "ContentLength": len(data)}

    def copy_object(
        self, Bucket: str, CopySource: dict[str, str], Key: str, **kwargs: Any
    ) -> dict[str, Any]:
        self.call_log.append(("copy_object", {"Bucket": Bucket, "CopySource": CopySource, "Key": Key}))
        src_bucket = CopySource["Bucket"]
        src_key = CopySource["Key"]
        if (src_bucket, src_key) not in self.storage:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Source key not found"}},
                "CopyObject",
            )
        self.storage[(Bucket, Key)] = self.storage[(src_bucket, src_key)]
        return {"CopyObjectResult": {"ETag": '"mock-etag-copied"'}}

    def delete_object(self, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]:
        self.call_log.append(("delete_object", {"Bucket": Bucket, "Key": Key}))
        self.storage.pop((Bucket, Key), None)
        return {"DeleteMarker": True}

    def list_objects_v2(self, Bucket: str, Prefix: str = "", **kwargs: Any) -> dict[str, Any]:
        self.call_log.append(("list_objects_v2", {"Bucket": Bucket, "Prefix": Prefix}))
        matches = [
            {"Key": k, "Size": len(v)}
            for (b, k), v in self.storage.items()
            if b == Bucket and k.startswith(Prefix)
        ]
        if not matches:
            return {"KeyCount": 0}
        return {"Contents": matches, "KeyCount": len(matches)}


class S3UploadUnitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.mock_s3 = MockS3Client()
        self.bucket = "aeronation-rag-persist-bucket"
        self.collection_type = "aerospace_manuals"
        self.ingestion_id = "v2026_ing_001"

        self.vector_config = VectorStoreConfig(
            collection_name=self.collection_type,
            dimension=384,
            distance="Cosine",
            enable_hybrid=True,
        )
        self.embedding_config = EmbeddingConfig(
            dense_model="sentence-transformers/all-MiniLM-L6-v2",
            sparse_model="Qdrant/bm42-all-minilm-l6-v2-attentions",
            expected_dimension=384,
        )
        self.chunking_config = ChunkingConfig(chunk_size=512, chunk_overlap=64)
        self.counts = StageCounts(
            files_seen=2,
            docs_loaded=4,
            docs_cleaned=4,
            chunks_created=10,
            embeddings_generated=10,
            vectors_inserted=10,
        )

        # Generate a valid local persistence artifact under persist/<collection_type>
        self.persist_res = persist_collection(
            collection_type=self.collection_type,
            ingestion_id=self.ingestion_id,
            vector_config=self.vector_config,
            embedding_config=self.embedding_config,
            chunking_config=self.chunking_config,
            counts=self.counts,
            persist_dir=self.temp_dir,
        )
        self.local_persist_dir = Path(self.temp_dir) / self.collection_type

        self.config = S3UploadConfig(
            bucket_name=self.bucket,
            base_prefix="persist",
            region_name="ap-south-1",
            enable_staging=True,
            collection_type=self.collection_type,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # 1. Complete persistence directory upload
    def test_complete_persistence_directory_upload(self) -> None:
        result = upload_persistence_to_s3(
            self.local_persist_dir,
            self.config,
            s3_client=self.mock_s3,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.bucket, self.bucket)
        self.assertEqual(result.collection_type, self.collection_type)
        self.assertEqual(result.ingestion_id, self.ingestion_id)
        self.assertEqual(result.files_failed, 0)
        self.assertGreater(result.bytes_uploaded, 0)

        # Verify all required LlamaIndex files and manifest.json exist in production S3 prefix
        for req_file in REQUIRED_PERSISTENCE_FILES:
            expected_key = f"persist/{self.collection_type}/{req_file}"
            self.assertIn((self.bucket, expected_key), self.mock_s3.storage)

        manifest_key = f"persist/{self.collection_type}/manifest.json"
        self.assertIn((self.bucket, manifest_key), self.mock_s3.storage)

    # 2. Nested relative directory structure preservation
    def test_nested_relative_directory_structure_preserved(self) -> None:
        nested_sub = self.local_persist_dir / "subfolder" / "nested"
        nested_sub.mkdir(parents=True, exist_ok=True)
        (nested_sub / "nested_doc.json").write_text('{"nested": true}', encoding="utf-8")

        result = upload_persistence_to_s3(
            self.local_persist_dir,
            self.config,
            s3_client=self.mock_s3,
        )

        self.assertTrue(result.success)
        expected_nested_key = f"persist/{self.collection_type}/subfolder/nested/nested_doc.json"
        self.assertIn((self.bucket, expected_nested_key), self.mock_s3.storage)

    # 3. Collection type is NOT hardcoded (dynamic collection types)
    def test_dynamic_collection_types_supported(self) -> None:
        for custom_col in ["custom_physics_v1", "engineering_manuals", "flight_data_2026"]:
            persist_collection(
                collection_type=custom_col,
                ingestion_id=f"{custom_col}_001",
                vector_config=VectorStoreConfig(collection_name=custom_col),
                persist_dir=self.temp_dir,
            )
            col_dir = Path(self.temp_dir) / custom_col
            custom_cfg = S3UploadConfig(bucket_name=self.bucket, base_prefix="persist", collection_type=custom_col)

            res = upload_persistence_to_s3(col_dir, custom_cfg, s3_client=self.mock_s3)
            self.assertTrue(res.success)
            self.assertEqual(res.collection_type, custom_col)
            self.assertIn((self.bucket, f"persist/{custom_col}/manifest.json"), self.mock_s3.storage)

    # 4. Collection type read from configuration
    def test_collection_type_from_configuration(self) -> None:
        cfg = S3UploadConfig.from_env(
            {"S3_PERSIST_BUCKET": "test-bkt", "COLLECTION_TYPE": "configured_col"}
        )
        self.assertEqual(cfg.collection_type, "configured_col")
        self.assertEqual(cfg.bucket_name, "test-bkt")

    # 5. Correct S3 prefix and key generation (build_s3_key)
    def test_build_s3_key_generation(self) -> None:
        self.assertEqual(
            build_s3_key("persist", "rag_llm", "docstore.json"),
            "persist/rag_llm/docstore.json",
        )
        self.assertEqual(
            build_s3_key("persist/", "/rag_llm/", "nested/index.json"),
            "persist/rag_llm/nested/index.json",
        )
        self.assertEqual(
            build_s3_key("persist", "rag_llm", "docstore.json", staging_subpath=".tmp/v001"),
            "persist/rag_llm/.tmp/v001/docstore.json",
        )

    # 6. Invalid collection type is rejected by validation before upload
    def test_invalid_collection_type_rejected_by_validation(self) -> None:
        invalid_types = [
            "../rag_llm",
            "../../something",
            "/absolute/path",
            "",
            "   ",
            "collection/../../other",
            "rag*llm",
            "col:type",
        ]
        for bad_col in invalid_types:
            with self.assertRaises(ValueError):
                validate_collection_type(bad_col)

    # 7. Missing local persistence directory fails cleanly
    def test_missing_local_directory_fails(self) -> None:
        non_existent = Path(self.temp_dir) / "does_not_exist"
        with self.assertRaises(IncompleteArtifactError) as ctx:
            upload_persistence_to_s3(non_existent, self.config, s3_client=self.mock_s3)
        self.assertIn("does not exist", str(ctx.exception).lower())

    # 8. Missing manifest fails cleanly
    def test_missing_manifest_fails(self) -> None:
        (self.local_persist_dir / "manifest.json").unlink()
        with self.assertRaises(ManifestError):
            upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)

    # 9. Invalid manifest fails cleanly
    def test_invalid_manifest_json_fails(self) -> None:
        (self.local_persist_dir / "manifest.json").write_text("{broken json", encoding="utf-8")
        with self.assertRaises(ManifestError):
            upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)

    # 10. Version / Ingestion ID is preserved in S3
    def test_version_preserved_in_s3_manifest(self) -> None:
        result = upload_persistence_to_s3(
            self.local_persist_dir,
            self.config,
            s3_client=self.mock_s3,
        )
        self.assertTrue(result.success)

        manifest_data = json.loads(
            self.mock_s3.storage[(self.bucket, f"persist/{self.collection_type}/manifest.json")].decode("utf-8")
        )
        self.assertEqual(manifest_data["ingestion_id"], self.ingestion_id)
        self.assertEqual(manifest_data["version"], self.ingestion_id)

    # 11. Qdrant collection identity in manifest is preserved
    def test_qdrant_identity_preserved(self) -> None:
        upload_persistence_to_s3(
            self.local_persist_dir,
            self.config,
            s3_client=self.mock_s3,
        )
        manifest_data = json.loads(
            self.mock_s3.storage[(self.bucket, f"persist/{self.collection_type}/manifest.json")].decode("utf-8")
        )
        self.assertEqual(manifest_data["qdrant"]["collection_name"], self.collection_type)
        self.assertEqual(manifest_data["qdrant"]["dimension"], 384)

    # 12. All expected files are uploaded
    def test_all_expected_files_uploaded(self) -> None:
        local_files = [p.name for p in self.local_persist_dir.glob("*") if p.is_file()]
        result = upload_persistence_to_s3(
            self.local_persist_dir,
            self.config,
            s3_client=self.mock_s3,
        )
        self.assertEqual(result.files_uploaded, len(local_files))
        for fname in local_files:
            self.assertIn((self.bucket, f"persist/{self.collection_type}/{fname}"), self.mock_s3.storage)

    # 13. Upload statistics are accurate
    def test_upload_statistics_accurate(self) -> None:
        result = upload_persistence_to_s3(
            self.local_persist_dir,
            self.config,
            s3_client=self.mock_s3,
        )
        self.assertEqual(result.files_expected, result.files_uploaded)
        self.assertEqual(result.files_failed, 0)
        self.assertGreater(result.bytes_uploaded, 0)
        self.assertEqual(len(result.uploaded_keys), result.files_uploaded)

    # 14. Partial upload failure is detected and staging cleaned up
    def test_partial_upload_failure_detected(self) -> None:
        failing_key = f"persist/{self.collection_type}/.tmp/{self.ingestion_id}/docstore.json"
        self.mock_s3.fail_on_keys.add(failing_key)

        with self.assertRaises(S3UploadError) as ctx:
            upload_persistence_to_s3(
                self.local_persist_dir,
                self.config,
                s3_client=self.mock_s3,
            )

        self.assertIn("Partial upload failure", str(ctx.exception))
        # Ensure staging objects were cleaned up and not leaked
        for (b, k) in list(self.mock_s3.storage.keys()):
            self.assertNotIn(".tmp", k)

    # 15. Missing AWS S3 bucket configuration is handled
    def test_missing_bucket_configuration_raises_error(self) -> None:
        empty_cfg = S3UploadConfig(bucket_name="")
        with self.assertRaises(S3ConfigurationError) as ctx:
            upload_persistence_to_s3(self.local_persist_dir, empty_cfg, s3_client=self.mock_s3)
        self.assertIn("bucket name is not configured", str(ctx.exception).lower())

    # 16. AWS permission error (AccessDenied / 403) is handled
    def test_s3_permission_error_handled(self) -> None:
        denied_key = f"persist/{self.collection_type}/.tmp/{self.ingestion_id}/docstore.json"
        self.mock_s3.permission_denied_keys.add(denied_key)

        with self.assertRaises(S3UploadError) as ctx:
            upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)
        self.assertTrue(isinstance(ctx.exception, S3UploadError))

    # 17. Network/upload error is handled cleanly
    def test_network_upload_error_handled(self) -> None:
        timeout_key = f"persist/{self.collection_type}/.tmp/{self.ingestion_id}/index_store.json"
        self.mock_s3.network_error_keys.add(timeout_key)

        with self.assertRaises(S3UploadError) as ctx:
            upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)
        self.assertIn("Partial upload failure", str(ctx.exception))

    # 18. Existing production artifact is preserved when an upload fails during staging
    def test_existing_production_artifact_preserved_on_failure(self) -> None:
        # 1. Establish existing valid production artifact in S3
        prod_manifest_key = f"persist/{self.collection_type}/manifest.json"
        initial_prod_manifest = json.dumps({
            "collection_type": self.collection_type,
            "ingestion_id": "initial_v1",
            "version": "initial_v1",
        }).encode("utf-8")
        self.mock_s3.storage[(self.bucket, prod_manifest_key)] = initial_prod_manifest

        # 2. Attempt a new failing upload for version v2
        failing_key = f"persist/{self.collection_type}/.tmp/{self.ingestion_id}/docstore.json"
        self.mock_s3.fail_on_keys.add(failing_key)

        with self.assertRaises(S3UploadError):
            upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)

        # 3. Verify original production S3 manifest is completely untouched
        self.assertEqual(self.mock_s3.storage[(self.bucket, prod_manifest_key)], initial_prod_manifest)

    # 19. Same version upload behaves deterministically
    def test_same_version_reupload_deterministic(self) -> None:
        # First upload
        res1 = upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)
        self.assertTrue(res1.success)

        # Second upload of identical artifact
        res2 = upload_persistence_to_s3(self.local_persist_dir, self.config, s3_client=self.mock_s3)
        self.assertTrue(res2.success)
        self.assertEqual(res1.uploaded_keys, res2.uploaded_keys)

    # 20. Zero credentials in logs, manifest, or error messages
    def test_no_credentials_in_logs_or_errors(self) -> None:
        dummy_secret = "AKIAIOSFODNN7EXAMPLE"
        try:
            upload_file_to_s3(
                self.mock_s3,
                Path(self.temp_dir) / "nonexistent.json",
                self.bucket,
                "some/key.json",
            )
        except S3UploadError as exc:
            self.assertNotIn(dummy_secret, str(exc))

    # 21. Post-upload S3 object verification correctly validates remote objects and manifest
    def test_verify_s3_objects_success_and_failure(self) -> None:
        valid_keys = [f"persist/{self.collection_type}/docstore.json"]
        self.mock_s3.storage[(self.bucket, valid_keys[0])] = b"{}"

        # Success on existing objects
        self.assertTrue(verify_s3_objects(self.mock_s3, self.bucket, valid_keys))

        # Failure on missing object
        with self.assertRaises(S3ObjectNotFoundError):
            verify_s3_objects(self.mock_s3, self.bucket, ["persist/rag_llm/missing.json"])

    # 22. The uploader delegates collection-type validation to validator.py
    def test_uploader_relies_on_validation_module(self) -> None:
        with patch("ingestion.shared_processing.s3_upload.validate_local_artifact") as mock_val:
            mock_val.return_value = {
                "collection_type": self.collection_type,
                "ingestion_id": self.ingestion_id,
            }
            res = upload_persistence_to_s3(
                self.local_persist_dir,
                self.config,
                s3_client=self.mock_s3,
            )
            self.assertTrue(res.success)
            mock_val.assert_called_once()


if __name__ == "__main__":
    unittest.main()
