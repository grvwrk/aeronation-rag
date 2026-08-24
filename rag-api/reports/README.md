# RAG reports

Generate the two local dashboards with:

```powershell
python scripts/generate_reports.py
```

The command writes:

- `reports/evals_report.html` — answer quality, retrieval quality, citations,
  latency, token usage, throughput, averages, and per-case results.
- `reports/realtime_report.html` — request telemetry from `logs/latency.log`,
  `logs/query_token_usage.log`, and `logs/token_stream.log`, including average
  and p95 values, stage timings, token usage, and per-request details.

Open either generated HTML file directly in a browser. The reports are local
artifacts and do not send data to an external analytics service.