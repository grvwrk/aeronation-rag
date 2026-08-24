"""Run offline RAG evals from a labeled JSONL dataset and predictions JSONL."""

from __future__ import annotations

import argparse
import asyncio
import importlib
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import (
    EvaluationCase,
    EvaluationReport,
    Prediction,
    evaluate_case_with_llm_judge,
    evaluate_dataset,
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_judge(factory_spec: str):
    module_name, separator, factory_name = factory_spec.rpartition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("--judge-factory must use the form module:function")
    factory = getattr(importlib.import_module(module_name), factory_name)
    judge = factory()
    return asyncio.run(judge) if inspect.isawaitable(judge) else judge


async def _evaluate_with_judge(
    cases: list[EvaluationCase], predictions: dict[str, Prediction], judge
) -> list[EvaluationReport]:
    reports = []
    for case in cases:
        deterministic = evaluate_dataset([case], predictions)[0]
        semantic = await evaluate_case_with_llm_judge(
            case, predictions[case.query], judge
        )
        metrics = {**deterministic.metrics, **semantic.metrics}
        checks = {
            **deterministic.checks,
            **{name: value >= 0.5 for name, value in semantic.metrics.items()},
        }
        reports.append(EvaluationReport(case.query, metrics, checks))
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "evals" / "dataset.jsonl"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--judge-factory",
        help="Optional LLM factory in the form module:function; makes live judge calls.",
    )
    args = parser.parse_args()

    cases = [EvaluationCase.from_dict(row) for row in _read_jsonl(args.dataset)]
    predictions = {
        row["query"]: Prediction.from_dict(row)
        for row in _read_jsonl(args.predictions)
    }
    if args.judge_factory:
        judge = _load_judge(args.judge_factory)
        reports = asyncio.run(_evaluate_with_judge(cases, predictions, judge))
    else:
        reports = evaluate_dataset(cases, predictions)
    for report in reports:
        values = " ".join(
            f"{name}={value:.2f}" for name, value in report.metrics.items()
        )
        print(f"{'PASS' if report.passed else 'FAIL'} {report.query}: {values}")
    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())