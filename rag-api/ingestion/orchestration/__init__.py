"""
Aeronation RAG Ingestion Orchestration Package.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .system_pipeline import cleanup_staging_artifacts, run_system_pipeline
from .user_pipeline import run_user_pipeline

__all__ = [
    "cleanup_staging_artifacts",
    "run_system_pipeline",
    "run_user_pipeline",
]


