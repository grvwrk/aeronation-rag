"""
User Pipeline Orchestration Module.
"""

import sys
from pathlib import Path

# Ensure project root (rag-api) is on sys.path so top-level modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.shared_processing.user_pipeline import run_user_pipeline
from models import UserPipelineConfig, UserPipelineResult

__all__ = ["run_user_pipeline", "UserPipelineConfig", "UserPipelineResult"]

