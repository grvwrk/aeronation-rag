"""Run labeled cases through a project-specific live RAG runner."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import (
    EvaluationCase,
    EvaluationReport,
    Prediction,
    aggregate_metrics,
    compare_baseline,
    evaluate_case,
    evaluate_case_with_llm_judge,
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_factory(spec: str):
    module_name, separator, function_name = spec.rpartition(":")
    if not separator or not module_name or not function_name:
        raise ValueError("Factory must use the form module:function")
    return getattr(importlib.import_module(module_name), function_name)


async def _call(factory, case: EvaluationCase):
    result = factory(case)
    return await result if inspect.isawaitable(result) else result


def _prediction(value: Prediction | dict[str, Any]) -> Prediction:
    return value if isinstance(value, Prediction) else Prediction.from_dict(value)


def _write_jsonl(
    path: Path, cases: list[EvaluationCase], predictions: list[Prediction]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for case, prediction in zip(cases, predictions):
            row = {"query": case.query, **dataclasses.asdict(prediction)}
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


async def _run(args) -> int:
    cases = [EvaluationCase.from_dict(row) for row in _read_jsonl(args.dataset)]
    runner = _load_factory(args.runner_factory)
    predictions = [_prediction(await _call(runner, case)) for case in cases]
    reports = [
        evaluate_case(case, prediction)
        for case, prediction in zip(cases, predictions)
    ]

    if args.judge_factory:
        judge_factory = _load_factory(args.judge_factory)
        judged_reports: list[EvaluationReport] = []
        for case, prediction, report in zip(cases, predictions, reports):
            judge = await _call(judge_factory, case)
            semantic = await evaluate_case_with_llm_judge(case, prediction, judge)
            judged_reports.append(
                EvaluationReport(
                    report.query,
                    {**report.metrics, **semantic.metrics},
                    {
                        **report.checks,
                        **{
                            name: score >= 0.5
                            for name, score in semantic.metrics.items()
                        },
                    },
                    {**report.explanations, **semantic.explanations},
                )
            )
        reports = judged_reports

    _write_jsonl(args.output_predictions, cases, predictions)
    aggregate = aggregate_metrics(reports)
    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_baseline.write_text(
            json.dumps({"metrics": aggregate}, indent=2) + "\n", encoding="utf-8"
        )
    if args.baseline:
        baseline_data = json.loads(args.baseline.read_text(encoding="utf-8"))
        baseline_metrics = baseline_data.get("metrics", baseline_data)
        comparison = compare_baseline(aggregate, baseline_metrics)
        print("Baseline:", json.dumps(comparison, sort_keys=True))
        if not comparison["passed"]:
            return 1
    for report in reports:
        print(
            f"{'PASS' if report.passed else 'FAIL'} {report.query}: "
            f"{json.dumps(report.metrics, sort_keys=True)}"
        )
    return 0 if all(report.passed for report in reports) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "evals" / "dataset.jsonl"
    )
    parser.add_argument(
        "--runner-factory",
        required=True,
        help="Live runner factory in the form module:function",
    )
    parser.add_argument(
        "--output-predictions",
        type=Path,
        default=PROJECT_ROOT / "evals" / "live_predictions.jsonl",
    )
    parser.add_argument(
        "--judge-factory",
        help="Optional LLM judge factory in the form module:function",
    )
    parser.add_argument(
        "--baseline", type=Path, help="Aggregate baseline JSON to compare against"
    )
    parser.add_argument(
        "--save-baseline", type=Path, help="Write aggregate metrics as baseline JSON"
    )
    return asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())