"""Evaluation utilities for the RAG answer and retrieval pipeline."""

from .evaluator import (
    EvaluationCase,
    EvaluationReport,
    Prediction,
    aggregate_metrics,
    compare_baseline,
    evaluate_case,
    evaluate_case_with_llm_judge,
    evaluate_dataset,
)

__all__ = [
    "EvaluationCase",
    "EvaluationReport",
    "Prediction",
    "aggregate_metrics",
    "compare_baseline",
    "evaluate_case",
    "evaluate_case_with_llm_judge",
    "evaluate_dataset",
]