"""Run offline RAG evals from a labeled JSONL dataset and predictions JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import EvaluationCase, Prediction, evaluate_dataset


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=PROJECT_ROOT / "evals" / "dataset.jsonl"
    )
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    cases = [EvaluationCase.from_dict(row) for row in _read_jsonl(args.dataset)]
    predictions = {
        row["query"]: Prediction.from_dict(row)
        for row in _read_jsonl(args.predictions)
    }
    reports = evaluate_dataset(cases, predictions)
    for report in reports:
        values = " ".join(
            f"{name}={value:.2f}" for name, value in report.metrics.items()
        )
        print(f"{'PASS' if report.passed else 'FAIL'} {report.query}: {values}")
    return 0 if all(report.passed for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())