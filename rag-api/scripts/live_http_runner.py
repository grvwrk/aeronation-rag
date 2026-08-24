"""HTTP adapter for run_live_evals.py against the streaming RAG API."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import requests

from evals import EvaluationCase, Prediction


def _estimate_tokens(text: str) -> int:
    return len(text.split())


def _decode_events(line: str) -> list[dict[str, Any]]:
    """Decode one or more JSON response events received in one stream chunk."""
    text = line.removeprefix("data: ").strip()
    decoder = json.JSONDecoder()
    events = []
    offset = 0
    while offset < len(text):
        while offset < len(text) and text[offset].isspace():
            offset += 1
        if offset >= len(text):
            break
        payload, end = decoder.raw_decode(text, offset)
        events.append(payload)
        offset = end
    return events


def _latest_log_event(filename: str, event_name: str) -> dict[str, Any]:
    path = Path(__file__).resolve().parent.parent / "logs" / filename
    if not path.exists():
        return {}
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if event.get("event") == event_name:
            return event
    return {}


def run_case(case: EvaluationCase) -> Prediction:
    """Run one case through /v1/chat using RAG_API_URL from the environment."""
    base_url = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000/v1/chat")
    timeout = float(os.environ.get("RAG_API_TIMEOUT", "300"))
    started = time.perf_counter()
    response = requests.post(
        base_url,
        json={"chat_id": "eval", "query": case.query},
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=timeout,
    )
    response.raise_for_status()
    answer_parts: list[str] = []
    contexts: list[str] = []
    citations: list[str] = []
    first_token_ms: float | None = None
    token_chunks = 0
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        for payload in _decode_events(line):
            event_type = payload.get("type")
            if event_type == "tokens":
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                answer_parts.append(str(payload.get("text", "")))
                token_chunks += 1
            elif event_type == "answer":
                answer_parts = [str(payload.get("text", ""))]
            elif event_type == "context":
                raw_context = payload.get("text", "{}")
                parsed_context = json.loads(raw_context) if isinstance(raw_context, str) else raw_context
                for item in parsed_context.values() if isinstance(parsed_context, dict) else []:
                    chunk = item.get("chunk")
                    if chunk:
                        contexts.append(str(chunk))
                    file_name = item.get("file_name")
                    if file_name:
                        citations.append(str(file_name))
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    answer = "".join(answer_parts)
    input_tokens = _estimate_tokens(case.query)
    output_tokens = _estimate_tokens(answer)
    retrieval = _latest_log_event("retrieval.log", "retrieval_result")
    request_complete = _latest_log_event("latency.log", "request_complete")
    request_id = request_complete.get("request_id") or retrieval.get("request_id")
    return Prediction(
        answer=answer,
        contexts=tuple(contexts),
        citations=tuple(dict.fromkeys(citations)),
        fallback_used=not contexts and not citations,
        request_id=request_id,
        retrieved_context_ids=tuple(retrieval.get("retrieved_context_ids", ())),
        retrieved_scores=tuple(float(score) for score in retrieval.get("retrieved_scores", ())),
        retrieved_nodes=retrieval.get("retrieved_nodes"),
        reranked_nodes=retrieval.get("reranked_nodes"),
        latency_ms=latency_ms,
        time_to_first_token_ms=first_token_ms,
        generation_duration_ms=latency_ms,
        input_tokens_estimate=input_tokens,
        output_tokens_estimate=output_tokens,
        total_tokens_estimate=input_tokens + output_tokens,
        output_tokens_per_second=round(output_tokens / (latency_ms / 1000), 2) if latency_ms else 0,
        token_chunks=token_chunks,
    )