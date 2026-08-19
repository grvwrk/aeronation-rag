"""
System Pipeline Orchestration Module.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.shared_processing.system_pipeline import (
    cleanup_staging_artifacts,
    main,
    run_system_pipeline,
)
from models import (
    IngestionState,
    IngestionStatus,
    PipelineConfig,
    PipelineResult,
)

__all__ = [
    "cleanup_staging_artifacts",
    "run_system_pipeline",
    "main",
    "PipelineConfig",
    "PipelineResult",
    "IngestionState",
    "IngestionStatus",
]


if __name__ == "__main__":
    main()
