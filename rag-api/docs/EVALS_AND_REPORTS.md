# Evals and reports

This document explains how to evaluate the RAG system, inspect runtime
performance, regenerate the dashboards, and contribute new evaluation cases.
It is written as a starting point for interns and developers who are new to
this repository.

## Why this exists

RAG quality has two dimensions:

1. **Answer quality**: Is the answer correct, supported by retrieved context,
   and properly cited?
2. **Operational quality**: Is the request fast enough, and is token usage and
   streaming behavior reasonable?

The project keeps these concerns together in one prediction format, then
renders them in two local HTML reports:

- `reports/evals_report.html` evaluates labeled examples.
- `reports/realtime_report.html` summarizes telemetry from actual API runs.

The reports are generated artifacts. The source data and report code are the
parts contributors should edit.

## Repository layout

| Path | Purpose |
| --- | --- |
| `evals/evaluator.py` | Evaluation models, metrics, thresholds, and optional LLM judge |
| `evals/dataset.jsonl` | Labeled questions and expected answers/contexts |
| `evals/predictions.jsonl` | Predictions and telemetry for the sample cases |
| `scripts/run_evals.py` | Command-line runner for evaluation results |
| `scripts/generate_reports.py` | Generates both HTML dashboards |
| `reports/evals_report.html` | Generated evaluation dashboard |
| `reports/realtime_report.html` | Generated runtime telemetry dashboard |
| `logs/latency.log` | Structured stage latency events |
| `logs/query_token_usage.log` | Per-request token and generation telemetry |
| `logs/token_stream.log` | Per-stream-chunk timing and token estimates |
| `tests/test_evals.py` | Deterministic tests for the evaluation package |

## Setup

Use the repository virtual environment on Windows:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& ".\.venv\Scripts\Activate.ps1"
```

Confirm that the selected interpreter is the project environment:

```powershell
python -c "import sys; print(sys.executable)"
```

The path should end with `.venv\Scripts\python.exe`.

## Running the evals

Run the deterministic tests:

```powershell
python tests/test_evals.py -v
```

Run the sample dataset through the CLI:

```powershell
python scripts/run_evals.py --predictions evals/predictions.jsonl
```

The CLI matches rows by the exact `query` string. It exits with status `0`
when every case passes and status `1` when a case fails or a prediction is
missing.

The Make target is equivalent:

```powershell
make evals
```

## How scoring works

The deterministic evaluator is intentionally dependency-light and does not
call an external model.

### Answer correctness

`answer_correctness` uses token-set F1 between the generated answer and the
reference answer. It is useful for catching clearly wrong or incomplete
answers, but it is not a substitute for human review or semantic judging.

### Groundedness

`groundedness` measures how many answer tokens also appear in the supplied
retrieved contexts. A low score usually means the answer contains unsupported
claims or the prediction did not include the contexts used to generate it.

### Context recall

`context_recall` checks whether each reference context has a sufficiently
similar match among the retrieved contexts. A low score points toward retrieval,
chunking, metadata filtering, or reranking problems.

### Citation coverage

`citation_coverage` is the fraction of expected citation identifiers present in
the prediction. Citation identifiers should use the same source naming used by
the application, such as `b787_overview.pdf`.

### Performance and token metrics

When present, the report also displays:

- `latency_ms`: total request latency
- `stage_latencies_ms`: stage-level latency map, such as retrieval and generation
- `time_to_first_token_ms`: delay before the first streamed token
- `generation_duration_ms`: generation duration
- `input_tokens_estimate`: estimated prompt/input tokens
- `output_tokens_estimate`: estimated generated tokens
- `total_tokens_estimate`: input plus output token estimate
- `output_tokens_per_second`: generation throughput
- `token_chunks`: number of streamed chunks
- `average_inter_chunk_ms`: average delay between chunks
- `max_inter_chunk_ms`: largest delay between chunks

These are estimates where the application says `estimate`; provider billing
numbers remain authoritative.

## Adding an evaluation case

Add one JSON object per line to [evals/dataset.jsonl](../evals/dataset.jsonl):

```json
{
  "query": "What does Mach number compare?",
  "reference_answer": "Mach number is the ratio of an object's speed to the speed of sound.",
  "reference_contexts": [
    "Mach number is speed of sound ratio. It determines compressible flow regime."
  ],
  "expected_citations": ["fluid_mechanics.pdf"],
  "max_latency_ms": 5000,
  "max_time_to_first_token_ms": 3000,
  "max_total_tokens": 600,
  "min_output_tokens_per_second": 1
}
```

Guidelines for good cases:

- Use questions real users are likely to ask.
- Keep the reference answer factual and concise.
- Copy reference contexts from the indexed source material.
- Include the expected source filename or citation identifier.
- Add performance budgets only after observing several real runs.
- Include edge cases such as no-result queries, metadata filters, and questions
  requiring more than one context when those behaviors matter.

For every dataset case, add a matching row to
[evals/predictions.jsonl](../evals/predictions.jsonl) while developing the
fixture. The query must match exactly.

## Prediction format

A minimal prediction is:

```json
{
  "query": "What does Mach number compare?",
  "answer": "Mach number is the ratio of an object's speed to the speed of sound.",
  "contexts": ["Mach number is speed of sound ratio."],
  "citations": ["fluid_mechanics.pdf"]
}
```

For a real run, add the telemetry fields listed above. The evaluator checks
token accounting when all three token estimates are present:

```text
total_tokens_estimate = input_tokens_estimate + output_tokens_estimate
```

If a required performance field is missing, its configured budget is not
evaluated. This allows quality-only cases and older prediction fixtures to
remain valid.

## Generating the reports

Generate both dashboards:

```powershell
python scripts/generate_reports.py
```

Or:

```powershell
make reports
```

The script reads:

- The evaluation dataset and predictions for the eval dashboard.
- `logs/latency.log` for stage latency events.
- `logs/query_token_usage.log` for request-level token telemetry.
- `logs/token_stream.log` for streamed chunk counts and timing.

It writes:

- `reports/evals_report.html`
- `reports/realtime_report.html`

Open either file directly in a browser. Regenerate the files after changing
the dataset, predictions, or logs. Do not manually edit generated HTML; change
the generator or its source data instead.

## Connecting a live RAG run

The production answer path is asynchronous and streams JSON events from
`Generate.generate_answer()`. A live adapter should collect those events into a
`Prediction`:

1. Concatenate the `text` values from `type == "tokens"` events.
2. Use the final `type == "answer"` event as the cleaned answer when available.
3. Parse the `type == "context"` event into the `contexts` field.
4. Extract citation identifiers from the returned context metadata or answer.
5. Attach request timing and token telemetry from the structured logs.
6. Pass the result to `evaluate_case()`.

The evaluator does not construct S3, Qdrant, Tavily, or LLM clients. Keeping
that integration in a small adapter makes the scoring code deterministic and
keeps local tests fast.

## Optional LLM judging

For semantic evaluation, use `evaluate_case_with_llm_judge()` from
[evals/evaluator.py](../evals/evaluator.py) with the configured LlamaIndex LLM:

```python
import asyncio

from evals import evaluate_case_with_llm_judge

report = asyncio.run(evaluate_case_with_llm_judge(case, prediction, llm))
```

The judge is asked to return JSON scores from `0` to `1` for:

- `correctness`
- `groundedness`
- `relevance`

This is opt-in because it makes a provider request and consumes quota. Treat
LLM judge scores as a review signal, not as an unquestionable source of truth.
Use deterministic tests for regressions and human review for important changes.

## Improving the system with the reports

Use the metric pattern to choose where to investigate:

| Symptom | Likely area to inspect |
| --- | --- |
| Low answer correctness, good context recall | QA/refine prompt or model behavior |
| Low groundedness | Prompt grounding, citation formatting, or unsupported generation |
| Low context recall | Chunking, embeddings, metadata filters, Qdrant retrieval, or reranking |
| Low citation coverage | Citation prompt, source metadata, or context post-processing |
| High retrieval latency | Qdrant, embeddings, reranker, or network path |
| High time to first token | LLM request, prompt size, or model cold start |
| High total token count | Prompt history, context size, or generation limits |
| Low output tokens per second | Provider/model load, streaming path, or local resource limits |
| Large max inter-chunk delay | Provider streaming pauses or event-loop contention |

Change one important variable at a time, rerun the same eval set, and compare
the generated report. A quality improvement that causes a significant latency
or token regression should be treated as a tradeoff and documented in the
change description.

## Contribution checklist

Before opening a pull request:

1. Add or update a representative dataset case.
2. Add a matching prediction fixture or live-run adapter output.
3. Run `python tests/test_evals.py -v`.
4. Run `python scripts/run_evals.py --predictions evals/predictions.jsonl`.
5. Run `python scripts/generate_reports.py` and inspect both HTML files.
6. Confirm that quality, latency, and token changes are intentional.
7. Update this document when the evaluation schema or report workflow changes.

Keep cases small, reproducible, and tied to a user-visible behavior. The goal
is not to maximize a single score; it is to make regressions obvious and give
the next developer enough evidence to improve the system confidently.