"""Run the 50 held-out questions through all three pipelines and package the outputs.

    python eval2/run_hidden.py                        # resumes, then packages
    python eval2/run_hidden.py --pipelines rag        # one pipeline
    python eval2/run_hidden.py --fresh                # start over
    python eval2/run_hidden.py --package              # rebuild the bundle, run nothing

Writes two files:

  data/results/hidden.jsonl     one row per (question, pipeline), full trace kept
  data/results/submission.json  the bundle for the organisers

We have no answers for these, so nothing here is scored. What is submitted is
what the organisers asked for: for every question, the answer from each of the
three pipelines with the tokens it cost and how long it took, plus the agentic
trace.
The trace is written in full rather than summarised -- the point of submitting it
is that someone else can audit how the answer was reached, and a summary written
by the system being audited is worth little.

Every row is stamped with the backend and model that produced it, and the
bundle's own labels are derived from the rows (eval2/provenance.py). The bundle
therefore says where its answers came from, not what the environment happened to
be set to when it was last written.

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

from agentic.config import HIDDEN_QUESTIONS, RESULTS_DIR, graph_backend_name, llm_model_name
from eval2.provenance import check_resumable, now, of_rows, stamp

RESULTS_FILE = RESULTS_DIR / "hidden.jsonl"
SUBMISSION_FILE = RESULTS_DIR / "submission.json"
SYSTEM = "Agentic GraphRAG over TigerGraph"
# Cheapest baseline first, so the bundle reads in the order the comparison is
# argued. Also the order rows are sorted within a question.
PIPELINE_ORDER = ("rag", "graphrag", "agentic")


def load_done(path: Path = RESULTS_FILE) -> dict[tuple[str, str], dict]:
    """Rows from previous runs, keyed by (qid, pipeline); later rows win.

    Rows written before this script ran more than one pipeline carry no
    `pipeline` field. They can only have come from the agent, because that is
    the only pipeline the script could run, so they are read as agentic rather
    than discarded -- a re-run of those answers would change nothing.
    """
    if not path.exists():
        return {}
    done = {}
    for line in path.open(encoding="utf-8"):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("status") != "error":
            row.setdefault("pipeline", "agentic")
            done[(row["qid"], row["pipeline"])] = row
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


def row_from_result(question: dict, result: dict, pipeline: str) -> dict:
    """One submission row: the answer, what it cost, and how it was reached."""
    row = {
        "qid": question["qid"],
        "pipeline": pipeline,
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
        # Of the wall clock, time slept on the provider's rate limit.
        "throttled_s": result.get("throttled_s", 0.0),
        "coverage_expected": result.get("coverage_expected"),
        "coverage_actual": result.get("coverage_actual"),
        "trace": trace_rows(result),
    }
    if result.get("error"):
        # Kept so a failed row says why. A resume retries it either way.
        row["error"] = result["error"]
    return row


def totals_of(rows: list[dict]) -> dict:
    """What one pipeline cost over the questions it answered."""
    answered = [r for r in rows if r.get("status") == "ok"]
    n = max(len(rows), 1)

    def tok(row: dict, field: str) -> int:
        return row.get("tokens", {}).get(field, 0)

    return {
        "questions": len(rows),
        "answered": len(answered),
        "refused": len(rows) - len(answered),
        "total_tokens": sum(tok(r, "total") for r in rows),
        "avg_tokens": round(sum(tok(r, "total") for r in rows) / n, 1),
        "avg_context_tokens": round(sum(tok(r, "context") for r in rows) / n, 1),
        "avg_steps": round(sum(len(r.get("trace") or []) for r in rows) / n, 2),
        # The organisers asked for latency per question; the throttled share is
        # kept beside it so the number is not read as compute time.
        "avg_latency_s": round(sum(r.get("wall_clock_s", 0.0) for r in rows) / n, 2),
        "avg_throttled_s": round(sum(r.get("throttled_s", 0.0) for r in rows) / n, 2),
    }


def bundle(rows: list[dict]) -> dict:
    """The submission bundle, labelled from its rows and nothing else.

    Totals are per pipeline. One average across all three would describe no
    system that exists, and the comparison between them is the point.
    """
    provenance = of_rows(rows)
    # Same reading as load_done: a row without a pipeline predates this script
    # running more than one, so it can only be the agent's.
    rows = [{**r, "pipeline": r.get("pipeline", "agentic")} for r in rows]
    names = [p for p in PIPELINE_ORDER if any(r["pipeline"] == p for r in rows)]
    return {
        "system": SYSTEM,
        "model": provenance["model"],
        "backend": provenance["backend"],
        # When the last answer was written, and when this file was assembled
        # from the rows. They differ whenever the bundle is rebuilt.
        "run_finished_at": provenance["run_finished_at"],
        "packaged_at": now(),
        "pipelines": names,
        "totals": {p: totals_of([r for r in rows if r["pipeline"] == p]) for p in names},
        "answers": sorted(rows, key=lambda r: (r["qid"], PIPELINE_ORDER.index(r["pipeline"]))),
    }


def package(results_file: Path = RESULTS_FILE, out: Path = SUBMISSION_FILE) -> dict:
    rows = list(load_done(results_file).values())
    built = bundle(rows)
    out.write_text(json.dumps(built, indent=2, ensure_ascii=False), encoding="utf-8")
    return built


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", action="store_true", help="discard previous rows")
    ap.add_argument("--package", action="store_true",
                    help="rebuild submission.json from hidden.jsonl without running")
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINE_ORDER),
                    choices=list(PIPELINE_ORDER),
                    help="which pipelines to run (default: all three)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only the first N questions (a smoke test before the full run)")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        RESULTS_FILE.unlink(missing_ok=True)

    questions = [json.loads(line) for line in HIDDEN_QUESTIONS.open(encoding="utf-8")]
    if args.limit:
        questions = questions[:args.limit]
    done = load_done()
    todo = [] if args.package else [(q, p) for q in questions for p in args.pipelines
                                    if (q["qid"], p) not in done]
    backend, model = graph_backend_name(), llm_model_name()
    started = time.perf_counter()

    if todo:
        # Rows already in the file must come from the same place these will.
        check_resumable(done.values(), backend, model)
        # Imported here so that packaging needs neither a graph nor an API key.
        from agentic.llm import LLMClient, QuotaExhausted
        from agentic.pipelines import agentic, graphrag, rag
        from agentic.tools.tools import Tools

        pipelines = {"rag": rag, "graphrag": graphrag, "agentic": agentic}
        tools, llm = Tools(), LLMClient()
        print(f"backend={backend} model={llm.settings.model}")
        print(f"{len(questions)} hidden questions x {len(args.pipelines)} pipelines "
              f"= {len(questions) * len(args.pipelines)} runs, {len(todo)} to run\n")
        with RESULTS_FILE.open("a", encoding="utf-8") as out:
            for index, (question, name) in enumerate(todo, 1):
                try:
                    result = pipelines[name].run(question["question"], tools, llm,
                                                 qid=question["qid"]).as_dict()
                except QuotaExhausted as exc:
                    print(f"\nstopping: {exc}\nresume with: python eval2/run_hidden.py")
                    break
                except Exception as exc:
                    result = {"qid": question["qid"], "question": question["question"],
                              "answer": "", "status": "error",
                              "error": f"{type(exc).__name__}: {exc}", "trace": []}
                row = stamp(row_from_result(question, result, name), backend, model)
                out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                out.flush()
                print(f"[{index:>3}/{len(todo)}] {question['qid']:<10}{name:<10}"
                      f"{question.get('qtype', ''):<13}"
                      f"{str(row['answer'])[:36]:<38}{row['tokens']['total']:>6}tok")

    built = package()
    print(f"\nran in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"{'pipeline':<10}{'answered':>10}{'refused':>9}{'avg tok':>9}"
          f"{'avg ctx':>9}{'avg s':>8}")
    for name, totals in built["totals"].items():
        print(f"{name:<10}{totals['answered']:>4}/{totals['questions']:<5}"
              f"{totals['refused']:>9}{totals['avg_tokens']:>9}"
              f"{totals['avg_context_tokens']:>9}{totals['avg_latency_s']:>8}")
    print(f"backend={built['backend']} model={built['model']} "
          f"run_finished_at={built['run_finished_at']}")
    print(f"wrote {RESULTS_FILE} and {SUBMISSION_FILE}")


if __name__ == "__main__":
    main()
