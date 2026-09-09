"""
Aggregate several eval runs into mean / min-max per pipeline, so reported numbers
reflect the run-to-run variance of an LLM judge instead of a single lucky (or
unlucky) sample. 55 questions x a nondeterministic judge means any one run moves
a few points; the honest headline is the mean with the observed range.

Usage:
    # after each run, keep a copy:  cp eval/results/eval_results.csv eval/results/run1.csv
    python eval/aggregate_runs.py eval/results/run1.csv eval/results/run2.csv eval/results/run3.csv
"""
import sys

import pandas as pd


def main(paths: list[str]):
    if len(paths) < 2:
        print("Give at least 2 run CSVs. Usage: python eval/aggregate_runs.py run1.csv run2.csv [...]")
        sys.exit(1)

    per_run = []
    for p in paths:
        df = pd.read_csv(p)
        errors = int((df["judge_source"] == "error").sum())
        stats = {}
        for name, sub in df.groupby("pipeline"):
            stats[name] = {
                "pass_pct": (sub["judge"] == "PASS").mean() * 100,
                "avg_tokens": sub["total_tokens"].mean(),
            }
        rag = stats.get("basic_rag", {}).get("avg_tokens") or 0
        gr = stats.get("graphrag", {}).get("avg_tokens") or 0
        stats["_reduction_pct"] = (1 - gr / rag) * 100 if rag else 0.0
        stats["_judge_errors"] = errors
        per_run.append((p, stats))

    pipelines = ["llm_only", "basic_rag", "graphrag"]
    print(f"Aggregating {len(per_run)} runs:\n")
    for p, s in per_run:
        line = "  ".join(f"{n}={s[n]['pass_pct']:.1f}%" for n in pipelines if n in s)
        print(f"  {p}: {line}  reduction={s['_reduction_pct']:.1f}%  judge_errors={s['_judge_errors']}")

    print("\n=== MEAN over runs (report these) ===")
    for n in pipelines:
        vals = [s[n]["pass_pct"] for _, s in per_run if n in s]
        toks = [s[n]["avg_tokens"] for _, s in per_run if n in s]
        if vals:
            print(f"  {n}: pass {sum(vals)/len(vals):.1f}%  "
                  f"(range {min(vals):.1f}-{max(vals):.1f})  "
                  f"avg_tokens {sum(toks)/len(toks):.0f}")
    reds = [s["_reduction_pct"] for _, s in per_run]
    print(f"  token_reduction: {sum(reds)/len(reds):.1f}%  (range {min(reds):.1f}-{max(reds):.1f})")
    total_err = sum(s["_judge_errors"] for _, s in per_run)
    if total_err:
        print(f"\nWARNING: {total_err} judge-error rows across runs (counted FAIL). "
              f"Re-run those rows or note this in the report.")


if __name__ == "__main__":
    main(sys.argv[1:])
