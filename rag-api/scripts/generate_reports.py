"""Generate local HTML dashboards for evals and real-time RAG telemetry."""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evals import EvaluationCase, Prediction, evaluate_case


LOG_JSON_PATTERN = re.compile(r"INFO (\{.*\})\s*$")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_log_events(path: Path) -> list[dict[str, Any]]:
    events = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        match = LOG_JSON_PATTERN.search(line)
        if match:
            try:
                events.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    return events


def _average(values: list[float]) -> float:
    return round(mean(values), 2) if values else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(len(ordered) * 0.95) - 1)
    return round(ordered[index], 2)


def _stats(values: list[float]) -> dict[str, float]:
    return {"average": _average(values), "p95": _p95(values), "count": len(values)}


def _metric_cards(stats: dict[str, Any]) -> str:
    return "".join(
        f'<article class="kpi"><span>{html.escape(str(label))}</span>'
        f'<strong>{html.escape(str(value))}</strong></article>'
        for label, value in stats.items()
    )


def _render_dashboard(
    title: str,
    subtitle: str,
    cards: dict[str, Any],
    charts: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> str:
    chart_markup = "".join(
        f'<section class="panel"><h2>{html.escape(chart["title"])}</h2>'
        f'<canvas id="{html.escape(chart["id"])}" height="220"></canvas></section>'
        for chart in charts
    )
    row_markup = ""
    if rows:
        headers = list(rows[0])
        row_markup = (
            '<section class="panel table-panel"><h2>Details</h2><div class="table-wrap">'
            '<table><thead><tr>'
            + "".join(f"<th>{html.escape(str(header))}</th>" for header in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(
                    f'<td>{html.escape(str(row.get(header, "")))}</td>'
                    for header in headers
                )
                + "</tr>"
                for row in rows
            )
            + "</tbody></table></div></section>"
        )
    data = json.dumps(charts, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>
:root {{ color-scheme: light; --ink:#17202a; --muted:#617080; --line:#d9e0e7; --paper:#f5f7f9; --panel:#fff; --teal:#087f8c; --orange:#d97736; --green:#39824d; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font:15px/1.5 Georgia,serif; }}
main {{ max-width:1240px; margin:auto; padding:42px 24px 64px; }} header {{ border-bottom:1px solid var(--line); padding-bottom:24px; margin-bottom:24px; }}
h1 {{ font:700 clamp(28px,4vw,46px)/1.05 Georgia,serif; margin:0 0 9px; letter-spacing:0; }} h2 {{ font:700 18px/1.2 Georgia,serif; margin:0 0 16px; }}
.subtitle {{ color:var(--muted); margin:0; }} .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:18px; }}
.kpi,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:6px; }} .kpi {{ padding:17px; }} .kpi span {{ display:block; color:var(--muted); font:12px/1.2 Arial,sans-serif; text-transform:uppercase; letter-spacing:.04em; }}
.kpi strong {{ display:block; color:var(--teal); font:700 25px/1.1 Arial,sans-serif; margin-top:9px; }} .charts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(360px,1fr)); gap:18px; }}
.panel {{ padding:20px; margin-bottom:18px; }} canvas {{ display:block; width:100%; height:220px; }} .table-wrap {{ overflow-x:auto; }} table {{ width:100%; border-collapse:collapse; font:13px/1.35 Arial,sans-serif; }}
th,td {{ text-align:left; padding:10px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }} th {{ color:var(--muted); font-size:11px; text-transform:uppercase; }}
@media(max-width:650px) {{ main {{ padding:26px 14px 42px; }} .charts {{ grid-template-columns:1fr; }} .panel {{ padding:14px; }} }}
</style></head><body><main><header><h1>{html.escape(title)}</h1><p class="subtitle">{html.escape(subtitle)}</p></header>
<section class="kpis">{_metric_cards(cards)}</section><section class="charts">{chart_markup}</section>{row_markup}</main>
<script>
const charts={data};
function draw(chart) {{ const canvas=document.getElementById(chart.id), ctx=canvas.getContext('2d');
  const dpr=window.devicePixelRatio||1, width=canvas.clientWidth, height=220; canvas.width=width*dpr; canvas.height=height*dpr; ctx.scale(dpr,dpr);
  const pad={{left:42,right:12,top:15,bottom:42}}, plotW=width-pad.left-pad.right, plotH=height-pad.top-pad.bottom;
  const max=Math.max(...chart.values,1), step=plotW/Math.max(chart.values.length,1); ctx.font='11px Arial'; ctx.fillStyle='#617080'; ctx.strokeStyle='#d9e0e7';
  [0,.5,1].forEach(t=>{{const y=pad.top+plotH*(1-t);ctx.beginPath();ctx.moveTo(pad.left,y);ctx.lineTo(width-pad.right,y);ctx.stroke();ctx.fillText((max*t).toFixed(0),4,y+4);}});
  chart.values.forEach((value,i)=>{{const barW=Math.max(12,step*.58), x=pad.left+i*step+(step-barW)/2, h=plotH*value/max, y=pad.top+plotH-h;ctx.fillStyle=chart.color;ctx.fillRect(x,y,barW,h);ctx.fillStyle='#17202a';ctx.save();ctx.translate(x+barW/2,height-pad.bottom+14);ctx.rotate(-.35);ctx.textAlign='center';ctx.fillText(String(chart.labels[i]).slice(0,22),0,0);ctx.restore();ctx.fillText(String(value),x+barW/2-12,y-5);}});
}}
charts.forEach(draw); window.addEventListener('resize',()=>charts.forEach(draw));
</script></body></html>"""


def build_evals_report(dataset_path: Path, predictions_path: Path, output_path: Path) -> None:
    cases = [EvaluationCase.from_dict(row) for row in _read_jsonl(dataset_path)]
    predictions = {
        row["query"]: Prediction.from_dict(row) for row in _read_jsonl(predictions_path)
    }
    reports = [evaluate_case(case, predictions[case.query]) for case in cases]
    metric_names = sorted({name for report in reports for name in report.metrics})
    averages = {name: _average([report.metrics[name] for report in reports if name in report.metrics]) for name in metric_names}
    quality_names = ["answer_correctness", "groundedness", "context_recall", "citation_coverage"]
    performance_names = ["latency_ms", "time_to_first_token_ms", "total_tokens_estimate", "output_tokens_per_second"]
    charts = []
    for chart_id, title, names, color in (
        ("quality", "Average quality scores", quality_names, "#087f8c"),
        ("performance", "Average runtime metrics", performance_names, "#d97736"),
    ):
        charts.append({"id": chart_id, "title": title, "labels": names, "values": [averages.get(name, 0) for name in names], "color": color})
    rows = []
    for report in reports:
        rows.append({"query": report.query, **{name: round(value, 2) for name, value in report.metrics.items()}, "status": "PASS" if report.passed else "FAIL"})
    cards = {"Cases": len(reports), "Passed": sum(report.passed for report in reports), "Avg correctness": averages.get("answer_correctness", 0), "Avg groundedness": averages.get("groundedness", 0), "Avg latency (ms)": averages.get("latency_ms", 0), "Avg output tok/s": averages.get("output_tokens_per_second", 0)}
    output_path.write_text(_render_dashboard("RAG evaluation report", "Quality and performance across the labeled evaluation set", cards, charts, rows), encoding="utf-8")


def build_realtime_report(log_dir: Path, output_path: Path) -> None:
    latency_events = _read_log_events(log_dir / "latency.log")
    token_events = _read_log_events(log_dir / "query_token_usage.log")
    stream_events = _read_log_events(log_dir / "token_stream.log")
    stage_values: dict[str, list[float]] = {}
    for event in latency_events:
        if event.get("stage") and isinstance(event.get("duration_ms"), (int, float)):
            stage_values.setdefault(str(event["stage"]), []).append(float(event["duration_ms"]))
    latency_values = [value for values in stage_values.values() for value in values]
    token_fields = ["input_tokens_estimate", "output_tokens_estimate", "total_tokens_estimate", "output_tokens_per_second", "generation_duration_ms", "time_to_first_token_ms"]
    token_averages = {field: _average([float(event[field]) for event in token_events if isinstance(event.get(field), (int, float))]) for field in token_fields}
    charts = [
        {"id": "stages", "title": "Average latency by stage (ms)", "labels": sorted(stage_values), "values": [_average(stage_values[name]) for name in sorted(stage_values)], "color": "#087f8c"},
        {"id": "tokens", "title": "Average token and generation telemetry", "labels": ["input", "output", "total", "tok/s", "generation ms", "TTFT ms"], "values": [token_averages[field] for field in token_fields], "color": "#d97736"},
    ]
    cards = {"Latency events": len(latency_events), "Token runs": len(token_events), "Stream chunks": len(stream_events), "Avg stage latency (ms)": _average(latency_values), "P95 stage latency (ms)": _p95(latency_values), "Avg total tokens": token_averages["total_tokens_estimate"], "Avg TTFT (ms)": token_averages["time_to_first_token_ms"], "P95 TTFT (ms)": _p95([float(event["time_to_first_token_ms"]) for event in token_events if isinstance(event.get("time_to_first_token_ms"), (int, float))]), "Avg output tok/s": token_averages["output_tokens_per_second"]}
    rows = [{"request_id": event.get("request_id", "unknown"), **{field: event.get(field, "") for field in token_fields}} for event in token_events]
    output_path.write_text(_render_dashboard("RAG real-time report", "Observed request latency, streaming, and token telemetry from local logs", cards, charts, rows), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "reports")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "logs")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build_evals_report(PROJECT_ROOT / "evals" / "dataset.jsonl", PROJECT_ROOT / "evals" / "predictions.jsonl", args.output_dir / "evals_report.html")
    build_realtime_report(args.log_dir, args.output_dir / "realtime_report.html")
    print(f"Generated {args.output_dir / 'evals_report.html'}")
    print(f"Generated {args.output_dir / 'realtime_report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())