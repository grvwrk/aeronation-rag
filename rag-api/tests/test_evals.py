"""Deterministic tests for the RAG evaluation contract."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import (
    EvaluationCase,
    Prediction,
    evaluate_case,
    evaluate_case_with_llm_judge,
    evaluate_dataset,
)


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = EvaluationCase(
            query="What is the 787 primary structure made from?",
            reference_answer="Composite materials make up 50 percent of the primary structure.",
            reference_contexts=("Composite materials make up 50 percent of the primary structure.",),
            expected_citations=("manual.pdf",),
        )

    def test_grounded_correct_answer_with_citation_passes(self) -> None:
        report = evaluate_case(
            self.case,
            Prediction(
                answer="Composite materials make up 50 percent of the primary structure.",
                contexts=self.case.reference_contexts,
                citations=("manual.pdf",),
            ),
        )

        self.assertTrue(report.passed)
        self.assertEqual(report.metrics["citation_coverage"], 1.0)

    def test_unsupported_answer_fails_groundedness(self) -> None:
        report = evaluate_case(
            self.case,
            Prediction(
                answer="The primary structure is made from titanium.",
                contexts=self.case.reference_contexts,
                citations=("manual.pdf",),
            ),
        )

        self.assertLess(report.metrics["groundedness"], 0.5)
        self.assertFalse(report.passed)

    def test_latency_and_token_telemetry_are_reported_and_checked(self) -> None:
        case = EvaluationCase(
            query="Telemetry",
            reference_answer="A grounded answer",
            max_latency_ms=2000,
            max_time_to_first_token_ms=1000,
            max_total_tokens=100,
            min_output_tokens_per_second=5,
        )
        prediction = Prediction(
            answer="A grounded answer",
            contexts=("A grounded answer",),
            latency_ms=1500,
            stage_latencies_ms={"retrieval": 300, "generation_end": 1500},
            time_to_first_token_ms=700,
            generation_duration_ms=1200,
            input_tokens_estimate=20,
            output_tokens_estimate=10,
            total_tokens_estimate=30,
            output_tokens_per_second=8,
            token_chunks=2,
            average_inter_chunk_ms=40,
            max_inter_chunk_ms=60,
        )

        report = evaluate_case(case, prediction)

        self.assertTrue(report.passed)
        self.assertEqual(report.metrics["total_tokens_estimate"], 30.0)
        self.assertEqual(report.metrics["latency_retrieval_ms"], 300.0)
        self.assertTrue(report.checks["token_accounting"])

    def test_dataset_preserves_case_order_and_requires_predictions(self) -> None:
        cases = [self.case, EvaluationCase("Second", "An answer")]
        predictions = {
            self.case.query: Prediction("Composite materials make up 50 percent of the primary structure."),
            "Second": Prediction("An answer"),
        }

        reports = evaluate_dataset(cases, predictions)

        self.assertEqual([report.query for report in reports], [case.query for case in cases])
        with self.assertRaises(KeyError):
            evaluate_dataset(cases, {self.case.query: predictions[self.case.query]})

    def test_llm_judge_scores_json_response(self) -> None:
        class FakeResponse:
            text = '{"correctness": 0.9, "groundedness": 0.8, "relevance": 1}'

        class FakeLLM:
            async def acomplete(self, prompt: str) -> FakeResponse:
                self.prompt = prompt
                return FakeResponse()

        import asyncio

        report = asyncio.run(
            evaluate_case_with_llm_judge(self.case, Prediction("An answer"), FakeLLM())
        )

        self.assertEqual(report.metrics["llm_groundedness"], 0.8)
        self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()