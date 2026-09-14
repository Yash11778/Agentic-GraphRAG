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
from eval2.score import cost_of_agency, summarise

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


def trim_trace(raw: dict) -> list[dict]:
    return [{f: step.get(f) for f in TRACE_FIELDS}
            for step in (raw.get("trace") or [])]


def main() -> None:
    public = latest(PUBLIC, lambda r: (r["qid"], r["pipeline"]))
    hidden = latest(HIDDEN, lambda r: r["qid"])
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
        "summary": summary,
        "cost_of_agency": {
            "vs_graphrag": cost_of_agency(summary),
            "vs_rag": cost_of_agency(summary, baseline="rag"),
        },
        "questions": sorted(questions.values(), key=lambda q: q["qid"]),
        "hidden": {
            "n": len(hidden),
            "answered": sum(1 for r in hidden if r.get("status") == "ok"),
            "total_tokens": sum(r.get("tokens", {}).get("total", 0) for r in hidden),
            "routes": _count(hidden, "route"),
            "stop_reasons": _count(hidden, "stop_reason"),
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    size_kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({size_kb:.0f} KB) "
          f"from {len(public)} public rows and {len(hidden)} hidden rows")


if __name__ == "__main__":
    main()
