"""Turn the result files into one JSON the dashboard reads.

    python eval2/report.py

The dashboard is a static page: it fetches this file and renders it, with no
server and no database behind it. That keeps the demo reproducible -- whoever
opens the page sees exactly the numbers in the repository, not whatever a live
run happens to produce.

Per-question rows are included but trimmed: full traces live in the result files,
and the viewer only needs enough of one to follow the investigation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic.config import RESULTS_DIR
from eval2.provenance import of_rows
from eval2.score import _median, cost_of_agency, summarise

PUBLIC = RESULTS_DIR / "public.jsonl"
HIDDEN = RESULTS_DIR / "hidden.jsonl"
OUT = ROOT / "frontend/public/report.json"

TRACE_FIELDS = ("step_no", "agent", "tool", "retrieval_method", "args_digest",
                "status", "latency_ms", "context_tokens", "chunks_returned",
                "new_evidence_count", "strategy_change", "note")


def latest(path: Path, key) -> list[dict]:
    """Rows from a JSONL result file, later rows superseding earlier ones."""
    if not path.exists():
        return []
    rows: dict = {}
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[key(row)] = row
    return list(rows.values())


def _count(rows: list[dict], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key) or "unknown"
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _by_qtype(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for row in rows:
        entry = out.setdefault(row.get("qtype") or "unknown", {"n": 0, "answered": 0})
        entry["n"] += 1
        entry["answered"] += row.get("status") == "ok"
    return dict(sorted(out.items()))


def trim_trace(raw: dict) -> list[dict]:
    return [{f: step.get(f) for f in TRACE_FIELDS}
            for step in (raw.get("trace") or [])]


def main() -> None:
    public = latest(PUBLIC, lambda r: (r["qid"], r["pipeline"]))
    # Hidden rows are one per (question, pipeline) since the held-out set began
    # being run through all three. Rows written before that carry no `pipeline`
    # and can only be the agent's; keying on qid alone would now collapse the
    # three pipelines onto whichever row happened to be written last.
    hidden = latest(HIDDEN, lambda r: (r["qid"], r.get("pipeline", "agentic")))
    hidden_by_pipeline = {
        name: [r for r in hidden if r.get("pipeline", "agentic") == name]
        for name in ("rag", "graphrag", "agentic")
    }
    hidden_by_pipeline = {k: v for k, v in hidden_by_pipeline.items() if v}
    hidden_agentic = hidden_by_pipeline.get("agentic", [])
    summary = summarise(public)

    questions: dict[str, dict] = {}
    for row in public:
        entry = questions.setdefault(row["qid"], {
            "qid": row["qid"], "qtype": row["qtype"],
            "question": row["raw"].get("question", ""),
            "gold": row["gold"], "pipelines": {},
        })
        entry["pipelines"][row["pipeline"]] = {
            "answer": row["answer"],
            "exact": row["exact"],
            "status": row["status"],
            "route": row["route"],
            "stop_reason": row["stop_reason"],
            "total_tokens": row["total_tokens"],
            "context_tokens": row["context_tokens"],
            "grounding_f1": row["grounding_f1"],
            "citations": (row["raw"].get("citations") or [])[:20],
            "trace": trim_trace(row["raw"]),
        }

    report = {
        # Which backend and model each result set was produced on, read from
        # the rows themselves so the page cannot say otherwise.
        "provenance": {"public": of_rows(public), "hidden": of_rows(hidden)},
        "summary": summary,
        "cost_of_agency": {
            "vs_graphrag": cost_of_agency(summary),
            "vs_rag": cost_of_agency(summary, baseline="rag"),
        },
        "questions": sorted(questions.values(), key=lambda q: q["qid"]),
        "hidden": {
            # The agent's own numbers stay at the top level: this section is
            # read as "what the submitted system did", and the baselines are
            # beside it for cost comparison rather than mixed into it.
            "n": len(hidden_agentic),
            "backend": of_rows(hidden)["backend"],
            "answered": sum(1 for r in hidden_agentic if r.get("status") == "ok"),
            "total_tokens": sum(r.get("tokens", {}).get("total", 0) for r in hidden_agentic),
            "avg_steps": (round(sum(len(r.get("trace") or []) for r in hidden_agentic)
                                / len(hidden_agentic), 2) if hidden_agentic else None),
            # Median wall clock net of time slept on the rate limit, as for the
            # public set; rows from before `throttled_s` was recorded count in full.
            "median_active_s": (round(_median([
                max(r.get("wall_clock_s", 0.0) - r.get("throttled_s", 0.0), 0.0)
                for r in hidden_agentic]), 2) if hidden_agentic else None),
            "routes": _count(hidden_agentic, "route"),
            "stop_reasons": _count(hidden_agentic, "stop_reason"),
            "by_qtype": _by_qtype(hidden_agentic),
            # The held-out set has no gold answers, so the only comparison that
            # can be made across pipelines here is cost, not accuracy.
            "by_pipeline": {
                name: {
                    "n": len(rows),
                    "answered": sum(1 for r in rows if r.get("status") == "ok"),
                    "total_tokens": sum(r.get("tokens", {}).get("total", 0) for r in rows),
                    "avg_tokens": round(sum(r.get("tokens", {}).get("total", 0)
                                            for r in rows) / len(rows), 1),
                    "avg_context_tokens": round(sum(r.get("tokens", {}).get("context", 0)
                                                    for r in rows) / len(rows), 1),
                    "median_active_s": round(_median([
                        max(r.get("wall_clock_s", 0.0) - r.get("throttled_s", 0.0), 0.0)
                        for r in rows]), 2),
                }
                for name, rows in hidden_by_pipeline.items()
            },
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB) "
          f"from {len(public)} public rows and {len(hidden)} hidden rows")


if __name__ == "__main__":
    main()
