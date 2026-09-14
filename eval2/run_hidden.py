"""Run the 50 held-out questions through the agent and package the outputs.

    python eval2/run_hidden.py            # resumes
    python eval2/run_hidden.py --fresh

Writes two files:

  data/results/hidden.jsonl     one row per question, full trace kept
  data/results/submission.json  the bundle for the organisers

We have no answers for these, so nothing here is scored. What is submitted is
what the brief asks for: the answer, the tokens it cost, and the agentic trace.
The trace is written in full rather than summarised -- the point of submitting it
is that someone else can audit how the answer was reached, and a summary written
by the system being audited is worth little.

The submission format is not specified in the guidebook. Until the organisers say
otherwise this writes the obvious shape and keeps every field the brief names, so
reshaping it later is a rename rather than a rerun.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic.config import HIDDEN_QUESTIONS, RESULTS_DIR, graph_backend_name
from agentic.llm import LLMClient, QuotaExhausted
from agentic.pipelines import agentic
from agentic.tools.tools import Tools

RESULTS_FILE = RESULTS_DIR / "hidden.jsonl"
SUBMISSION_FILE = RESULTS_DIR / "submission.json"


def load_done() -> dict[str, dict]:
    if not RESULTS_FILE.exists():
        return {}
    done = {}
    for line in RESULTS_FILE.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") != "error":
            done[row["qid"]] = row
    return done


def trace_rows(result: dict) -> list[dict]:
    """The per-step record the brief asks for, straight from the run."""
    return [
        {
            "step": step["step_no"],
            "agent": step["agent"],
            "tool": step["tool"],
            "retrieval_method": step["retrieval_method"],
            "arguments": step["args_digest"],
            "status": step["status"],
            "latency_ms": step["latency_ms"],
            "context_tokens": step["context_tokens"],
            "chunks_returned": step["chunks_returned"],
            "new_evidence": step["new_evidence_count"],
            "strategy_change": step["strategy_change"],
            "note": step["note"],
        }
        for step in (result.get("trace") or [])
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        RESULTS_FILE.unlink(missing_ok=True)

    questions = [json.loads(line) for line in HIDDEN_QUESTIONS.open(encoding="utf-8")]
    done = load_done()
    todo = [q for q in questions if q["qid"] not in done]
    tools, llm = Tools(), LLMClient()

    print(f"backend={graph_backend_name()} model={llm.settings.model}")
    print(f"{len(questions)} hidden questions, {len(todo)} to run\n")

    started = time.perf_counter()
    with RESULTS_FILE.open("a", encoding="utf-8") as out:
        for index, question in enumerate(todo, 1):
            try:
                result = agentic.run(question["question"], tools, llm,
                                     qid=question["qid"]).as_dict()
            except QuotaExhausted as exc:
                print(f"\nstopping: {exc}\nresume with: python eval2/run_hidden.py")
                break
            except Exception as exc:
                result = {"qid": question["qid"], "question": question["question"],
                          "pipeline": "agentic", "answer": "", "status": "error",
                          "error": f"{type(exc).__name__}: {exc}", "trace": []}
            row = {
                "qid": question["qid"],
                "qtype": question.get("qtype", ""),
                "question": question["question"],
                "answer": result["answer"],
                "status": result["status"],
                "route": result.get("route", ""),
                "stop_reason": result.get("stop_reason", ""),
                "citations": result.get("citations", []),
                "tokens": {
                    "context": result.get("context_tokens", 0),
                    "input": result.get("input_tokens", 0),
                    "output": result.get("output_tokens", 0),
                    "total": result.get("total_tokens", 0),
                },
                "llm_calls": result.get("llm_calls", 0),
                "tool_calls": result.get("tool_calls", 0),
                "wall_clock_s": result.get("wall_clock_s", 0.0),
                "coverage_expected": result.get("coverage_expected"),
                "coverage_actual": result.get("coverage_actual"),
                "trace": trace_rows(result),
            }
            if result.get("error"):
                # Kept so a failed row says why. A resume retries it either way.
                row["error"] = result["error"]
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            out.flush()
            print(f"[{index:>2}/{len(todo)}] {question['qid']:<10}{question.get('qtype', ''):<13}"
                  f"{str(row['answer'])[:40]:<42}{row['tokens']['total']:>6}tok")

    rows = list(load_done().values())
    answered = [r for r in rows if r["status"] == "ok"]
    bundle = {
        "system": "Agentic GraphRAG over TigerGraph",
        "model": llm.settings.model,
        "backend": graph_backend_name(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "totals": {
            "questions": len(rows),
            "answered": len(answered),
            "refused": len(rows) - len(answered),
            "total_tokens": sum(r["tokens"]["total"] for r in rows),
            "avg_tokens": round(sum(r["tokens"]["total"] for r in rows) / max(len(rows), 1), 1),
            "avg_steps": round(sum(len(r["trace"]) for r in rows) / max(len(rows), 1), 2),
        },
        "answers": sorted(rows, key=lambda r: r["qid"]),
    }
    SUBMISSION_FILE.write_text(json.dumps(bundle, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(f"\n{len(rows)}/50 answered in {(time.perf_counter() - started) / 60:.1f} min"
          f"  ({bundle['totals']['refused']} refused)")
    print(f"wrote {RESULTS_FILE} and {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
