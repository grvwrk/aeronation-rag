"""Dependency-light evaluation primitives for RAG experiments and CI.

The evaluator deliberately accepts predictions instead of constructing the API's
S3, Qdrant, and model clients. A live runner can therefore be supplied by a
developer, while the scoring and regression checks remain deterministic.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(value.lower()))


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return " ".join(str(item) for item in value.values())
    return str(value)


@dataclass(frozen=True)
class EvaluationCase:
    """A labeled query used to evaluate one RAG response."""

    query: str
    reference_answer: str
    reference_contexts: tuple[str, ...] = ()
    expected_citations: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    max_latency_ms: float | None = None
    max_time_to_first_token_ms: float | None = None
    max_total_tokens: int | None = None
    min_output_tokens_per_second: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvaluationCase":
        return cls(
            query=str(value["query"]),
            reference_answer=str(value["reference_answer"]),
            reference_contexts=tuple(value.get("reference_contexts", ())),
            expected_citations=tuple(value.get("expected_citations", ())),
            metadata=value.get("metadata", {}),
            max_latency_ms=value.get("max_latency_ms"),
            max_time_to_first_token_ms=value.get("max_time_to_first_token_ms"),
            max_total_tokens=value.get("max_total_tokens"),
            min_output_tokens_per_second=value.get("min_output_tokens_per_second"),
        )


@dataclass(frozen=True)
class Prediction:
    """The model output needed by the offline RAG metrics."""

    answer: str
    contexts: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    latency_ms: float | None = None
    stage_latencies_ms: Mapping[str, float] = field(default_factory=dict)
    time_to_first_token_ms: float | None = None
    generation_duration_ms: float | None = None
    input_tokens_estimate: int | None = None
    output_tokens_estimate: int | None = None
    total_tokens_estimate: int | None = None
    output_tokens_per_second: float | None = None
    token_chunks: int | None = None
    average_inter_chunk_ms: float | None = None
    max_inter_chunk_ms: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Prediction":
        contexts = value.get("contexts", ())
        return cls(
            answer=str(value.get("answer", "")),
            contexts=tuple(_as_text(context) for context in contexts),
            citations=tuple(str(citation) for citation in value.get("citations", ())),
            latency_ms=value.get("latency_ms"),
            stage_latencies_ms=value.get("stage_latencies_ms", {}),
            time_to_first_token_ms=value.get("time_to_first_token_ms"),
            generation_duration_ms=value.get("generation_duration_ms"),
            input_tokens_estimate=value.get("input_tokens_estimate"),
            output_tokens_estimate=value.get("output_tokens_estimate"),
            total_tokens_estimate=value.get("total_tokens_estimate"),
            output_tokens_per_second=value.get("output_tokens_per_second"),
            token_chunks=value.get("token_chunks"),
            average_inter_chunk_ms=value.get("average_inter_chunk_ms"),
            max_inter_chunk_ms=value.get("max_inter_chunk_ms"),
        )


@dataclass(frozen=True)
class EvaluationReport:
    """Per-case metrics and their aggregate score."""

    query: str
    metrics: Mapping[str, float]
    checks: Mapping[str, bool] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) if self.checks else all(
            value >= 0.5 for value in self.metrics.values()
        )


def _f1(candidate: str, reference: str) -> float:
    candidate_tokens = _tokens(candidate)
    reference_tokens = _tokens(reference)
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)
    overlap = len(candidate_tokens & reference_tokens)
    precision = overlap / len(candidate_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _groundedness(answer: str, contexts: Iterable[str]) -> float:
    answer_tokens = _tokens(answer)
    context_tokens = _tokens(" ".join(contexts))
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & context_tokens) / len(answer_tokens)


def _context_recall(references: Iterable[str], contexts: Iterable[str]) -> float:
    references = list(references)
    contexts = list(contexts)
    if not references:
        return 1.0
    if not contexts:
        return 0.0
    recalled = sum(
        any(_f1(context, reference) >= 0.5 for context in contexts)
        for reference in references
    )
    return recalled / len(references)


def _citation_coverage(expected: Iterable[str], actual: Iterable[str]) -> float:
    expected = set(expected)
    actual = set(actual)
    if not expected:
        return 1.0
    return len(expected & actual) / len(expected)


def evaluate_case(case: EvaluationCase, prediction: Prediction) -> EvaluationReport:
    """Score one response using deterministic metrics suitable for CI."""

    metrics = {
        "answer_correctness": _f1(prediction.answer, case.reference_answer),
        "groundedness": _groundedness(prediction.answer, prediction.contexts),
        "context_recall": _context_recall(
            case.reference_contexts, prediction.contexts
        ),
        "citation_coverage": _citation_coverage(
            case.expected_citations, prediction.citations
        ),
    }
    for name, value in prediction.stage_latencies_ms.items():
        metrics[f"latency_{name}_ms"] = float(value)
    optional_metrics = {
        "latency_ms": prediction.latency_ms,
        "time_to_first_token_ms": prediction.time_to_first_token_ms,
        "generation_duration_ms": prediction.generation_duration_ms,
        "input_tokens_estimate": prediction.input_tokens_estimate,
        "output_tokens_estimate": prediction.output_tokens_estimate,
        "total_tokens_estimate": prediction.total_tokens_estimate,
        "output_tokens_per_second": prediction.output_tokens_per_second,
        "token_chunks": prediction.token_chunks,
        "average_inter_chunk_ms": prediction.average_inter_chunk_ms,
        "max_inter_chunk_ms": prediction.max_inter_chunk_ms,
    }
    metrics.update({name: float(value) for name, value in optional_metrics.items() if value is not None})

    checks = {
        "answer_correctness": metrics["answer_correctness"] >= 0.5,
        "groundedness": metrics["groundedness"] >= 0.5,
        "context_recall": metrics["context_recall"] >= 0.5,
        "citation_coverage": metrics["citation_coverage"] >= 0.5,
    }
    if case.max_latency_ms is not None and prediction.latency_ms is not None:
        checks["max_latency_ms"] = prediction.latency_ms <= case.max_latency_ms
    if (
        case.max_time_to_first_token_ms is not None
        and prediction.time_to_first_token_ms is not None
    ):
        checks["max_time_to_first_token_ms"] = (
            prediction.time_to_first_token_ms <= case.max_time_to_first_token_ms
        )
    if case.max_total_tokens is not None and prediction.total_tokens_estimate is not None:
        checks["max_total_tokens"] = prediction.total_tokens_estimate <= case.max_total_tokens
    if (
        case.min_output_tokens_per_second is not None
        and prediction.output_tokens_per_second is not None
    ):
        checks["min_output_tokens_per_second"] = (
            prediction.output_tokens_per_second >= case.min_output_tokens_per_second
        )
    if (
        prediction.input_tokens_estimate is not None
        and prediction.output_tokens_estimate is not None
        and prediction.total_tokens_estimate is not None
    ):
        checks["token_accounting"] = prediction.total_tokens_estimate == (
            prediction.input_tokens_estimate + prediction.output_tokens_estimate
        )
    return EvaluationReport(query=case.query, metrics=metrics, checks=checks)


def evaluate_dataset(
    cases: Iterable[EvaluationCase], predictions: Mapping[str, Prediction]
) -> list[EvaluationReport]:
    """Evaluate cases by their exact query, preserving dataset order."""

    reports = []
    for case in cases:
        if case.query not in predictions:
            raise KeyError(f"Missing prediction for query: {case.query}")
        reports.append(evaluate_case(case, predictions[case.query]))
    return reports


async def evaluate_case_with_llm_judge(
    case: EvaluationCase, prediction: Prediction, llm: Any
) -> EvaluationReport:
    """Add semantic judge scores using any LlamaIndex-compatible LLM.

    The judge must return a JSON object with scores from 0 to 1 for
    ``correctness``, ``groundedness``, and ``relevance``. Keeping the adapter
    generic lets callers reuse the app's configured OpenAI, Groq, Anthropic, or
    Ollama model without coupling the evaluator to one provider.
    """

    prompt = (
        "Evaluate this RAG response. Return JSON only with numeric scores from 0 to 1 "
        "for correctness, groundedness, and relevance. Groundedness means every "
        "material claim is supported by the contexts.\n"
        f"Question: {case.query}\n"
        f"Reference answer: {case.reference_answer}\n"
        f"Answer: {prediction.answer}\n"
        f"Contexts: {list(prediction.contexts)}"
    )
    response = await llm.acomplete(prompt)
    raw_text = getattr(response, "text", str(response)).strip()
    try:
        scores = json.loads(raw_text)
        metrics = {
            "llm_correctness": float(scores["correctness"]),
            "llm_groundedness": float(scores["groundedness"]),
            "llm_relevance": float(scores["relevance"]),
        }
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("LLM judge returned invalid evaluation JSON") from exc

    if any(score < 0 or score > 1 for score in metrics.values()):
        raise ValueError("LLM judge scores must be between 0 and 1")
    return EvaluationReport(query=case.query, metrics=metrics)