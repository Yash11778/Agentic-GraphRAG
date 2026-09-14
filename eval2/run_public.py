"""Run the 100 public questions through all three pipelines and score them.

    python eval2/run_public.py                          # everything, resuming
    python eval2/run_public.py --limit 10               # a slice, for a smoke test
    python eval2/run_public.py --pipelines agentic      # one pipeline
    python eval2/run_public.py --fresh                  # ignore previous results
    python eval2/run_public.py --qids pub-004 pub-015   # named questions only
    python eval2/run_public.py --out data/results/check.jsonl --redo ...
                                                        # a side run, headline file untouched
    python eval2/run_public.py --compact                # drop superseded rows from the file

Results append to `data/results/public.jsonl`, one row per (question, pipeline),
and a rerun skips what is already there. That matters more than it looks: a
300-run benchmark against a rate-limited free tier will be interrupted, and
losing an hour of completed work to one timeout would push the whole schedule.

Failures are recorded as rows with `status="error"`, never dropped. A silently
shorter result file is how a benchmark starts flattering itself (LOCKED-6).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agentic.config import PUBLIC_QUESTIONS, RESULTS_DIR, graph_backend_name
from agentic.llm import LLMClient, QuotaExhausted
from agentic.pipelines import agentic, graphrag, rag
from agentic.tools.tools import Tools
from eval2.score import cost_of_agency, score_row, summarise

PIPELINES = {"rag": rag, "graphrag": graphrag, "agentic": agentic}
RESULTS_FILE = RESULTS_DIR / "public.jsonl"
SUMMARY_FILE = RESULTS_DIR / "public_summary.json"


def load_questions(limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in PUBLIC_QUESTIONS.open(encoding="utf-8")]
    return rows[:limit] if limit else rows


def load_done(path: Path, include_errors: bool = True) -> dict[tuple[str, str], dict]:
    """Rows from previous runs, keyed by (qid, pipeline); later rows win.

    `include_errors=False` treats a failed row as unfinished. An error row is an
    infrastructure failure, not a result, and a resume that skipped it would bake
    a provider outage into the benchmark permanently.
    """
    if not path.exists():
        return {}
    done = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a partial last line from an interrupted run
            if include_errors or row.get("status") != "error":
                done[(row["qid"], row["pipeline"])] = row
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pipelines", nargs="+", default=list(PIPELINES),
                    choices=list(PIPELINES))
    ap.add_argument("--fresh", action="store_true", help="ignore previous results")
    ap.add_argument("--qtypes", nargs="+", default=None,
                    help="only these question types")
    ap.add_argument("--redo", action="store_true",
                    help="rerun the selected slice even if results exist")
    ap.add_argument("--qids", nargs="+", default=None, help="only these question ids")
    ap.add_argument("--out", type=Path, default=RESULTS_FILE,
                    help="write rows here instead of the headline results file; "
                         "the summary is only rewritten for the headline file")
    ap.add_argument("--compact", action="store_true",
                    help="rewrite the results file keeping the latest row per "
                         "(question, pipeline), then exit")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file: Path = args.out
    if args.compact:
        compact(results_file)
        return
    if args.fresh:
        results_file.unlink(missing_ok=True)

    questions = load_questions(args.limit)
    if args.qtypes:
        questions = [q for q in questions if q.get("qtype") in args.qtypes]
    if args.qids:
        wanted = set(args.qids)
        questions = [q for q in questions if q["qid"] in wanted]
    done = {} if args.redo else load_done(results_file, include_errors=False)
    tools, llm = Tools(), LLMClient()

    todo = [(q, name) for q in questions for name in args.pipelines
            if (q["qid"], name) not in done]
    print(f"backend={graph_backend_name()} model={llm.settings.model} "
          f"credentials={len(llm.settings.all_keys)}")
    print(f"{len(questions)} questions x {len(args.pipelines)} pipelines "
          f"= {len(questions) * len(args.pipelines)} runs, {len(todo)} to do\n")

    started = time.perf_counter()
    with results_file.open("a", encoding="utf-8") as out:
        for index, (question, name) in enumerate(todo, 1):
            try:
                result = PIPELINES[name].run(question["question"], tools, llm,
                                             qid=question["qid"]).as_dict()
            except QuotaExhausted as exc:
                # Stop rather than fail the rest: every remaining question would
                # hit the same wall, and the file stays resumable as it is.
                print(f"\nstopping: {exc} (all {len(llm.settings.all_keys)} "
                      f"credentials spent)")
                print(f"resume with: python eval2/run_public.py "
                      f"({len(todo) - index + 1} runs left)")
                break
            except Exception as exc:
                result = {"qid": question["qid"], "question": question["question"],
                          "pipeline": name, "answer": "", "status": "error",
                          "error": f"{type(exc).__name__}: {exc}", "trace": []}
            row = score_row(result, question)
            row["raw"] = result
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            out.flush()
            mark = "ok " if row["exact"] else ("~  " if row["lenient"] else "   ")
            print(f"[{index:>3}/{len(todo)}] {mark}{question['qid']:<9}{name:<9}"
                  f"{str(row['answer'])[:34]:<36} gold={str(row['gold'])[:24]:<26}"
                  f"{row['total_tokens']:>6}tok")

    rows = list(load_done(results_file).values())
    summary = summarise(rows)
    if results_file == RESULTS_FILE:
        payload = {"summary": summary, "cost_of_agency": cost_of_agency(summary),
                   "n_rows": len(rows), "model": llm.settings.model,
                   "backend": graph_backend_name()}
        SUMMARY_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\nfinished in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"{'pipeline':<10}{'exact':>8}{'lenient':>9}{'tokens':>9}{'ground F1':>11}"
          f"{'steps':>7}{'median s':>10}")
    for name, stats in summary.items():
        print(f"{name:<10}{stats['exact']:>8.1%}{stats['lenient']:>9.1%}"
              f"{stats['avg_total_tokens']:>9.0f}{stats['avg_grounding_f1']:>11.3f}"
              f"{stats['avg_steps']:>7.1f}{stats['median_active_s']:>10.1f}")
    print(f"\nwrote {results_file}" + (f" and {SUMMARY_FILE}"
                                        if results_file == RESULTS_FILE else ""))


def compact(path: Path) -> None:
    """Rewrite a results file with only the latest row per (question, pipeline).

    Resumed runs append, so a file that has been through several sessions holds
    every superseded attempt. Readers already take the last row per key; this
    makes the file say what the readers see, and a 5MB file becomes 2MB.
    """
    rows = load_done(path)
    if not rows:
        print(f"nothing to compact in {path}")
        return
    before = sum(1 for _ in path.open(encoding="utf-8"))
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        for row in rows.values():
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)
    print(f"{path}: {before} rows -> {len(rows)}")


if __name__ == "__main__":
    main()
